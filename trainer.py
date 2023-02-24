import pandas as pd
import seaborn as sn
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.optim.lr_scheduler import *
from torch.optim.swa_utils import *
from torchvision import transforms
from torch.utils.data import Subset, ConcatDataset
from torch.optim import SGD
from med_data import MedImageFolders

from acquisition.batch_bald import *
from utils import *
from models import *
from transformed_dataset import TransformedDataset


# @class ATTrainTest
# @abstract
# @discussion

class ATTrainTest:
    def __init__(self, classes: list[str], model: str, criterion: str, optimizer: str, acquirer: str,
                 norm: transforms.Normalize, train_data: Dataset, val_data: Dataset = None, test_data: Dataset = None,
                 batch_size: int = 128, acquisition_batch_size: int = 10, resume: bool = True) -> None:

        # Dataloader initialization
        self._batchSize = 128
        if batch_size > 0:
            self._batchSize = batch_size

        if not train_data:
            raise AssertionError

        self._norm = norm
        self._trainTransform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self._norm,
        ])
        self._testTransform = transforms.Compose([
            transforms.ToTensor(),
            self._norm,
        ])

        self._cudaAvailable = torch.cuda.is_available()
        self.set_train_data(train_data, val_data)
        if test_data:
            self.set_test_data(test_data)

        self._classNames = classes
        self._numFeatures = len(classes)

        # Device & model initialization
        self._modelStr = model
        if model == "SimpleDLA":
            self._model = SimpleDLA(num_classes=self._numFeatures)
        elif model == "ResNet34":
            self._model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34')
            self._model.fc = nn.Linear(self._model.fc.in_features, self._numFeatures)  # For resnet
        self._device = torch.device('cuda' if self._cudaAvailable else 'cpu')
        self._model = self._model.to(self._device)
        if self._cudaAvailable:
            self._model = nn.DataParallel(self._model)
            cudnn.benchmark = True

        # Criterion, optimizer, scheduler
        if criterion == "CrossEntropyLoss":
            self._criterion = nn.CrossEntropyLoss()
        if optimizer == "SGD":
            self._optimizer = SGD(self._model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)

        self._scheduler = CosineAnnealingLR(self._optimizer, T_max=200)
        self._swaModel = None
        self._swaScheduler = SWALR(self._optimizer, anneal_strategy="cos", anneal_epochs=20, swa_lr=0.05)
        if acquirer == "BatchBALD":
            self._acquirer = ATBatchDisagreement(acquisition_batch_size, self._model)

        # Store the best accuracy & load the imported model if it exists
        self._bestAcc = 0.0
        self._minLoss = 100.0

        if resume:
            print("[+] Loading from checkpoint...")
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if os.path.isfile(self.__get_ckpt_path()):
                self._load_model()
                print("[+] Model loaded.")
            else:
                print("[!] No checkpoint available!")
            print()

        self._printProb = False
        self._graphCFM = False
        self._graphLoss = False
        self._confusionMatrix = torch.zeros(self._numFeatures, self._numFeatures)

        self._trainAUROC = []
        self._trainLoss = []
        self._trainAcc = []
        self._valAUROC = []
        self._valLoss = []
        self._valAcc = []

        self._slideProbs = {}

        self.set_visual_options(False, True, True)

    def __del__(self) -> None:
        self.get_best_acc()
        self.get_min_loss()

    def __get_ckpt_path(self) -> str:
        if self._modelStr == "SimpleDLA":
            return './checkpoint/dla_simple.pth'
        if self._modelStr == "ResNet34":
            return './checkpoint/resnet34.pth'

    def _save_model(self) -> None:
        state = {
            'weights': self._model.state_dict(),
            'optim': self._optimizer.state_dict(),
            'acc': self._bestAcc,
            'loss': self._minLoss,
        }
        torch.save(state, self.__get_ckpt_path())

    def _load_model(self) -> None:
        checkpoint = torch.load(self.__get_ckpt_path())
        self._model.load_state_dict(checkpoint['weights'])
        self._optimizer.load_state_dict(checkpoint['optim'])
        self._bestAcc = checkpoint['acc']
        self._minLoss = checkpoint['loss']

    def set_train_data(self, train_data: Dataset, val_data: Dataset) -> None:
        if not val_data:
            total_size = len(train_data)
            train_size = int(0.9 * total_size)
            val_size = total_size - train_size

            self._origTrainData = Subset(train_data, indices=torch.arange(total_size))
            self._trainData, self._valData = random_split(self._origTrainData, [train_size, val_size])
        else:
            self._origTrainData = ConcatDataset([train_data, val_data])
            self._trainData = train_data
            self._valData = val_data

        self._trainData = TransformedDataset(self._trainData, transformer=self._trainTransform)
        self._valData = TransformedDataset(self._valData, transformer=self._testTransform)

        self._trainLoader = DataLoader(self._trainData, batch_size=self._batchSize, shuffle=True,
                                       pin_memory=self._cudaAvailable, num_workers=4)
        self._valLoader = DataLoader(self._valData, batch_size=self._batchSize, shuffle=True,
                                     pin_memory=self._cudaAvailable, num_workers=4)

    def set_test_data(self, test_data: Dataset) -> None:
        self._origTestData = Subset(test_data, indices=torch.arange(len(test_data)))
        self._testData = TransformedDataset(self._origTestData, transformer=self._testTransform)
        self._testLoader = DataLoader(self._testData, batch_size=self._batchSize,
                                      pin_memory=self._cudaAvailable, num_workers=4, shuffle=True)

    def set_visual_options(self, prob: bool = False, cfm: bool = False, loss: bool = True):
        self._printProb = prob
        self._graphCFM = cfm
        self._graphLoss = loss

    def get_best_acc(self) -> float:
        print("[+] Model best accuracy: {:.4f}".format(self._bestAcc))
        return self._bestAcc

    def get_min_loss(self) -> float:
        print("[+] Model minimum loss: {:.4f}".format(self._minLoss))
        return self._minLoss

    def _plot_cfm(self):
        # Build confusion matrix
        plt.figure(figsize=(12, 7))
        df_cm = pd.DataFrame(self._confusionMatrix, index=self._classNames, columns=self._classNames).astype(int)
        heatmap = sn.heatmap(df_cm, annot=True, fmt="d")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

        # ROC-AUC; Consider replacing with roc_curve function
        if self._numFeatures == 2:
            tn, fp, fn, tp = self._confusionMatrix.ravel()
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)

            roc_values = [[tpr, fpr]]
            tpr_values, fpr_values = zip(*roc_values)
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.plot(fpr_values, tpr_values)
            ax.plot(np.linspace(0, 1, 100),
                    np.linspace(0, 1, 100),
                    label='baseline',
                    linestyle='--')
            plt.title('Receiver Operating Characteristic Curve', fontsize=18)
            plt.ylabel('TPR', fontsize=16)
            plt.xlabel('FPR', fontsize=16)
            plt.legend(fontsize=12)
            plt.show()

    def __train_val_once(self, num_images: int = 0, mode: int = 0) -> (float, float, int):
        # First set to eval mode
        if mode == 0:
            print("[-] Training...")
            self._model.train()
            loader = self._trainLoader
        else:
            print("[-] Evaluating...")
            self._model.eval()
            loader = self._valLoader if mode == 1 else self._testLoader
        assert loader, "[!] Dataloader is not allocated."

        num_images_left = num_images
        loss_sum = 0.0
        loss_cnt = 0
        correct_count = 0
        total = 0
        roc_auc = 0.0
        all_labels = []
        all_probs = []

        with torch.set_grad_enabled(mode == 0):
            for index, sample in enumerate(loader):
                # Parse inputs and labels
                inputs = sample[0].to(self._device)
                labels = sample[1].to(self._device)
                has_path = False
                if len(sample) > 2:
                    has_path = True
                    paths = sample[2]

                # Zero gradients only in train mode
                (mode == 0) and self._optimizer.zero_grad(set_to_none=True)

                # Calculate outputs and loss
                outputs = self._model(inputs) if ((mode == 0) or not self._swaModel) else self._swaModel(inputs)
                loss = self._criterion(outputs, labels)

                # Backward propagation & stepping optimizer
                if mode == 0:
                    loss.backward()
                    self._optimizer.step()

                # Stats calculation
                loss_sum += loss.item()
                predicted = torch.squeeze(outputs.argmax(dim=1, keepdim=True))
                total += labels.size(0)
                correct_count += predicted.eq(labels.view_as(predicted)).sum().item()

                all_labels.extend(labels.cpu().detach().numpy().tolist())
                prob = torch.softmax(outputs, dim=1).squeeze(1)
                if self._numFeatures == 2:
                    prob = prob[:, 1]
                prob = prob.cpu().detach().numpy().tolist()
                if has_path:
                    for i in range(len(prob)):
                        name = os.path.normpath(paths[i]).split(os.sep)[-2]
                        if name in self._slideProbs:
                            self._slideProbs[name].append(prob[i])
                        else:
                            self._slideProbs[name] = [prob[i]]
                all_probs.extend(prob)
                try:
                    if self._numFeatures == 2:
                        roc_auc = roc_auc_score(all_labels, all_probs)
                    else:
                        roc_auc = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovo',
                                                labels=np.arange(self._numFeatures))
                    if roc_auc < 0.5:
                        print("[!] ROC AUC Score is particularly low.")
                        print("[-] True labels: {}".format(labels))
                        print("[-] Predicted probability of positive class: {}".format(prob))
                except:
                    roc_auc = 0
                    print("[!] ROC AUC Score calculation failed!")

                progress_bar(index, len(loader), 'ROC AUC Score: %.3f | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (roc_auc, loss_sum / (index + 1), 100. * correct_count / total, correct_count, total))
                loss_cnt = index + 1

                # Visualization
                for i in range(inputs.size()[0]):
                    if num_images_left == 0:
                        break
                    if num_images > 1:
                        ax = plt.subplot(num_images // 2, 2, num_images_left)
                    else:
                        ax = plt.subplot(1, 1, 1)
                    ax.axis('off')
                    ax.set_title(f'Predicted: {self._classNames[predicted[i]]}')
                    imshow(inputs.cpu().data[i])
                    num_images_left -= 1
                if self._graphCFM:
                    for t, p in zip(labels.view(-1), predicted.view(-1)):
                        self._confusionMatrix[t.long(), p.long()] += 1

        print()
        return roc_auc, loss_sum / loss_cnt, correct_count / total

    def _train_val_once(self, epoch: int, num_epochs: int, num_images: int = 0):
        print()
        print('[*] Epoch {}/{}'.format(epoch, num_epochs))
        swa = (epoch > (num_epochs // 2))
        train_auroc, train_loss, train_acc = self.__train_val_once(num_images)
        if self._graphCFM:
            self._plot_cfm()
            self._confusionMatrix = torch.zeros(self._numFeatures, self._numFeatures)
        if swa:
            if not self._swaModel:
                self._swaModel = AveragedModel(self._model)
            self._swaModel.update_parameters(self._model)
        val_auroc, val_loss, val_acc = self.__train_val_once(num_images, 1)
        if self._graphCFM:
            self._plot_cfm()
            self._confusionMatrix = torch.zeros(self._numFeatures, self._numFeatures)

        self._swaScheduler.step() if swa else self._scheduler.step()

        print('[*] Training results:')
        print(
            '[*] ROC AUC Score: {:.4f} Loss: {:.4f} Acc: {:.4f}'.format(train_auroc, train_loss, train_acc))
        print('[*] Validation results:')
        print(
            '[*] ROC AUC Score: {:.4f} Loss: {:.4f} Acc: {:.4f}'.format(val_auroc, val_loss, val_acc))

        if self._slideProbs:
            for key in self._slideProbs:
                def get_avg(l):
                    return sum(l) / len(l)

                avg = get_avg(self._slideProbs[key])
                self._slideProbs[key] = [avg]

            print('[*] Positive probability for each slide:')
            for key in self._slideProbs:
                print('[+] Slide name: {}'.format(key))
                print('[+] Probability: {:.4f}'.format(self._slideProbs[key][0]))

        # Make a copy of the model if the accuracy on the validation set has improved
        if val_acc > self._bestAcc and val_loss < self._minLoss:
            self._bestAcc = val_acc
            self._minLoss = val_loss
            self._save_model()  # Now we'll load in the best model weights
            print("[+] Model updated.")
        # else:
        #    print("[!] Reverting model as there is no improvement in performance.")
        #    self._load_model()

        return [train_auroc, train_loss, train_acc * 100], [val_auroc, val_loss, val_acc * 100]

    def _plot_stats(self, train_res, val_res, append: bool = True, plot: bool = True):
        if append:
            self._trainAUROC.append(train_res[0])
            self._valAUROC.append(val_res[0])
            self._trainLoss.append(train_res[1])
            self._valLoss.append(val_res[1])
            self._trainAcc.append(train_res[2])
            self._valAcc.append(val_res[2])

        if plot:
            x_auroc = range(len(self._trainAUROC))
            plt.plot(x_auroc, self._trainAUROC, 'g', label="Training")
            plt.plot(x_auroc, self._valAUROC, 'r', label="Validation")
            plt.title("ROC AUC Score Curve")
            plt.show()

            x_loss = range(len(self._trainLoss))
            plt.plot(x_loss, self._trainLoss, 'g', label="Training")
            plt.plot(x_loss, self._valLoss, 'r', label="Validation")
            plt.title("Loss Curve")
            plt.show()

            x_acc = range(len(self._trainAcc))
            plt.plot(x_acc, self._trainAcc, 'g', label="Training")
            plt.plot(x_acc, self._valAcc, 'r', label="Validation")
            plt.title("Accuracy Curve")
            plt.show()

    def _reset_stats(self):
        self._trainAUROC = []
        self._trainLoss = []
        self._trainAcc = []
        self._valAUROC = []
        self._valLoss = []
        self._valAcc = []

    def train(self, num_epochs: int = 10, num_images: int = 0) -> (list, list):
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            train_res, val_res = self._train_val_once(epoch, num_epochs, num_images)
            if self._graphLoss:
                self._plot_stats(train_res, val_res, True, True)

        self._reset_stats()
        if self._swaModel:
            update_bn(self._trainLoader, self._swaModel)

        run_time = time.time() - start_time
        print('[+] Training completed in {:.0f}m {:.0f}s'.format(run_time // 60, run_time % 60))

    def train_active(self, num_images: int = 0) -> None:
        total_size = len(self._trainData)
        train_size = int(0.95 * total_size)
        pool_size = total_size - train_size

        train, pool = random_split(self._trainData, [train_size, pool_size])
        train = TransformedDataset(train, transformer=self._trainTransform)
        pool = TransformedDataset(pool, transformer=self._trainTransform)

        orig_len = len(pool)
        while len(pool) > 0:
            print()
            print(f'[*] Acquiring BatchBALD batch. Pool size: {len(pool)}')
            best_indices = self._acquirer.select_batch(pool, self._numFeatures)
            move_data(best_indices, pool, train)
            self._trainLoader = DataLoader(train, batch_size=self._batchSize, shuffle=True,
                                           pin_memory=self._cudaAvailable, num_workers=4)
            train_res, val_res = self._train_val_once(orig_len - len(pool), orig_len, num_images)
            if self._graphLoss:
                self._plot_stats(train_res, val_res, True, True)

        self._reset_stats()
        if self._swaModel:
            update_bn(self._trainLoader, self._swaModel)

    def train_cross_validate(self, k_folds: int = 5, folders: list[str] = None, num_images: int = 0):
        if folders:
            total_size = len(folders)
        else:
            total_size = len(self._origTrainData)

        fraction = 1 / k_folds
        seg = int(total_size * fraction)

        # tr: train, val: valid; r: right, l: left
        # [trll, trlr], [vall, valr], [trrl, trrr]
        for i in range(k_folds):
            print('[+] Fold {}'.format(i + 1))
            print('--------------------------------')
            train_left_right = i * seg
            val_left = train_left_right
            val_right = i * seg + seg
            train_right_left = val_right
            train_right_right = total_size

            train_left_indices = list(range(0, train_left_right))
            val_indices = list(range(val_left, val_right))
            train_right_indices = list(range(train_right_left, train_right_right))
            train_indices = train_left_indices + train_right_indices

            if folders:
                train_set = MedImageFolders([folders[i] for i in train_indices])
                val_set = MedImageFolders([folders[i] for i in val_indices])
            else:
                train_set = torch.utils.data.dataset.Subset(self._origTrainData, train_indices)
                val_set = torch.utils.data.dataset.Subset(self._origTrainData, val_indices)

            self._trainData = TransformedDataset(train_set, transformer=self._trainTransform)
            self._valData = TransformedDataset(val_set, transformer=self._testTransform)

            self._trainLoader = torch.utils.data.DataLoader(self._trainData, batch_size=self._batchSize, shuffle=True)
            self._valLoader = torch.utils.data.DataLoader(self._valData, batch_size=self._batchSize, shuffle=True)
            self.train_active(num_images)

    def test(self, num_images: int = 0) -> None:
        self.__train_val_once(num_images, 2)


# https://odsc.medium.com/crash-course-pool-based-sampling-in-active-learning-cb40e30d49df

# to-do: maybe plot loss/acc change in epoch?