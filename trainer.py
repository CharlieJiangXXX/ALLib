import pandas as pd
import seaborn as sn
import torch.backends.cudnn as cudnn
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import *
from torch.optim.swa_utils import *
from torchvision import transforms
from torch.utils.data import Subset, SubsetRandomSampler, ConcatDataset
from torch.optim import SGD

from acquisition.batch_bald import *
from utils import *
from models import *
from transformed_dataset import TransformedDataset


# @class ATTrainTest
# @abstract
# @discussion

class ATTrainTest:
    def __init__(self, classes: list[str], model: str, criterion: str, optimizer: str, acquirer: str,
                 norm: transforms.Normalize, train_data: Dataset, test_data: Dataset = None,
                 batch_size: int = 512, acquisition_batch_size: int = 4, resume: bool = True) -> None:

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
        self.set_train_data(train_data)
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
            self._model.fc = nn.Linear(model.fc.in_features, self._numFeatures)  # For resnet
        self._device = torch.device('cuda' if self._cudaAvailable else 'cpu')
        self._model = self._model.to(self._device)
        if self._cudaAvailable:
            self._model = nn.DataParallel(self._model)
            cudnn.benchmark = True

        # Criterion, optimizer, scheduler
        if criterion == "CrossEntropyLoss":
            self._criterion = nn.CrossEntropyLoss()
        if optimizer == "SGD":
            self._optimizer = SGD(self._model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

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

        self._printProb = False
        self._graphCFM = False
        self._graphLoss = False
        self._confusionMatrix = torch.zeros(self._numFeatures, self._numFeatures)

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
        if self._modelStr == "SimpleDLA":
            torch.save(state, self.__get_ckpt_path())

    def _load_model(self) -> None:
        checkpoint = torch.load(self.__get_ckpt_path())
        self._model.load_state_dict(checkpoint['weights'])
        self._optimizer.load_state_dict(checkpoint['optim'])
        self._bestAcc = checkpoint['acc']
        self._minLoss = checkpoint['loss']

    def set_train_data(self, train_data: Dataset) -> None:
        total_size = len(train_data)
        train_size = int(0.9 * total_size)
        val_size = total_size - train_size

        self._origTrainData = Subset(train_data, indices=torch.arange(total_size))
        self._trainRaw, self._valRaw = random_split(self._origTrainData, [train_size, val_size])

        train_transformed = TransformedDataset(self._trainRaw, transformer=self._trainTransform)
        val_transformed = TransformedDataset(self._valRaw, transformer=self._testTransform)

        self._trainLoader = DataLoader(train_transformed, batch_size=self._batchSize, shuffle=True,
                                       pin_memory=self._cudaAvailable, num_workers=4)
        self._valLoader = DataLoader(val_transformed, batch_size=self._batchSize,
                                     pin_memory=self._cudaAvailable, num_workers=4)

    def set_test_data(self, test_data: Dataset) -> None:
        self._origTestData = Subset(test_data, indices=torch.arange(len(test_data)))
        test_transformed = TransformedDataset(self._origTestData, transformer=self._testTransform)
        self._testLoader = DataLoader(test_transformed, batch_size=self._batchSize,
                                      pin_memory=self._cudaAvailable, num_workers=4)

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

        # ROC-AUC
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

    def _train_test_once(self, num_images: int = 0, mode: int = 0) -> (float, int):
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

        with torch.set_grad_enabled(mode == 0):
            for index, (inputs, labels) in enumerate(loader):
                # Parse inputs and labels
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)

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
                predicted = outputs.argmax(dim=1, keepdim=True)
                total += labels.size(0)
                correct_count += predicted.eq(labels.view_as(predicted)).sum().item()

                progress_bar(index, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (loss_sum / (index + 1), 100. * correct_count / total, correct_count, total))
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
                if self._printProb:
                    m = nn.Softmax(dim=1)
                    print(m(outputs))
                if self._graphCFM:
                    for t, p in zip(labels.view(-1), predicted.view(-1)):
                        self._confusionMatrix[t.long(), p.long()] += 1

        return loss_sum / loss_cnt, correct_count / total

    def train(self, num_epochs: int = 10, num_images: int = 0) -> (list, list):
        start_time = time.time()
        epoch_loss = []
        epoch_acc = []

        for epoch in range(1, num_epochs + 1):
            print()
            print('[*] Epoch {}/{}'.format(epoch, num_epochs))
            swa = (epoch > (num_epochs // 2))
            self._train_test_once(num_images)
            if swa:
                if not self._swaModel:
                    self._swaModel = AveragedModel(self._model)
                self._swaModel.update_parameters(self._model)
            loss, acc = self._train_test_once(num_images, 1)

            self._swaScheduler.step() if swa else self._scheduler.step()

            epoch_loss.append(loss)
            epoch_acc.append(acc * 100)
            print('[*] Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss[-1], epoch_acc[-1]))
            if self._graphCFM:
                self._plot_cfm()

            # Make a copy of the model if the accuracy on the validation set has improved
            if epoch_acc[-1] > self._bestAcc:
                self._bestAcc = epoch_acc[-1]
                self._minLoss = epoch_loss[-1]
                self._save_model()  # Now we'll load in the best model weights
                print("[+] Model updated.")

        update_bn(self._trainLoader, self._swaModel)
        if self._graphLoss:
            plt.plot(epoch_loss)
            plt.plot(epoch_acc)
            plt.show()

        run_time = time.time() - start_time
        print('[+] Training completed in {:.0f}m {:.0f}s'.format(run_time // 60, run_time % 60))
        return epoch_loss, epoch_acc

    def test(self, num_images: int = 0) -> None:
        self._train_test_once(num_images, 2)

    def train_crossval(self, k_folds: int = 5, num_images: int = 0):
        kfold = KFold(n_splits=k_folds, shuffle=True)
        for fold, (train_ids, test_ids) in enumerate(kfold.split(self._origTrainData)):
            print(f'[+] Fold {fold + 1}')
            print('--------------------------------')

            dataset = ConcatDataset([self._origTrainData, self._origTestData])
            dataset = TransformedDataset(dataset, transformer=self._trainTransform)

            train_subsampler = SubsetRandomSampler(train_ids)
            test_subsampler = SubsetRandomSampler(test_ids)

            self._trainLoader = torch.utils.data.DataLoader(dataset, batch_size=self._batchSize,
                                                            sampler=train_subsampler)
            self._valLoader = torch.utils.data.DataLoader(dataset, batch_size=self._batchSize, sampler=test_subsampler)

            self.train(1, num_images)
        self.test()

    def train_active(self, num_images: int = 0) -> None:
        total_size = len(self._trainRaw)
        train_size = int(0.8 * total_size)
        pool_size = total_size - train_size

        train, pool = random_split(self._trainRaw, [train_size, pool_size])
        train = TransformedDataset(train, transformer=self._trainTransform)
        pool = TransformedDataset(pool, transformer=self._trainTransform)

        while len(pool) > 0:
            print(f'Acquiring BatchBALD batch. Pool size: {len(pool)}')
            best_indices = self._acquirer.select_batch(pool)
            move_data(best_indices, pool, train)
            self._trainLoader = DataLoader(train, batch_size=self._batchSize, shuffle=True,
                                           pin_memory=self._cudaAvailable, num_workers=4)
            self.train(1, num_images)

        self.test(num_images)

# https://odsc.medium.com/crash-course-pool-based-sampling-in-active-learning-cb40e30d49df