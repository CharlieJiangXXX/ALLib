import os

import time
import pandas as pd
import seaborn as sn
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import *
from torch.optim.swa_utils import *
from torchvision import transforms
from torch.utils.data import Subset, ConcatDataset
from torch.optim import SGD, Adam
from med_data import MedImageFolders
from pathlib import Path

from acquisition.batch_bald import *
from utils import *
from models import *
from transformed_dataset import TransformedDataset


# @class ATTrainTest
# @abstract
# @discussion

class ATTrainTest:
    def __init__(self, name: str, classes: list[str], model: str, criterion: str, optimizer: str, acquirer: str,
                 norm: transforms.Normalize, train_data: Dataset, val_data: Dataset = None, test_data: Dataset = None,
                 batch_size: int = 128, acquisition_batch_size: int = 1, resume: bool = True) -> None:

        # Dataloader initialization
        self._batchSize = 128
        if batch_size > 0:
            self._batchSize = batch_size

        if not train_data:
            raise AssertionError

        self._norm = norm
        self._trainTransform = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
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
        self._set_model(name, model)

        # Criterion, optimizer, scheduler
        self._set_criterion(criterion)
        self._set_optimizer(optimizer)
        self._set_scheduler("SWA")
        self._set_acquirer(acquirer, acquisition_batch_size)

        # Store the best accuracy & load the imported model if it exists
        self._bestAcc = 0.0
        self._minLoss = 100.0

        self._ckptPath = self._origCkptPath

        if not os.path.isdir('outputs'):
            os.mkdir('outputs')

        if resume:
            print("[+] Loading from checkpoint...")
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if os.path.isfile(self._ckptPath):
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

        self._slideTrainProbs = {}
        self._slideValProbs = {}
        self._slideTestProbs = {}

        self.set_visual_options(False, True, True)

    def __del__(self) -> None:
        self.get_best_acc()
        self.get_min_loss()

    def _set_model(self, name, desc):
        if desc:
            self._modelDesc = desc
        elif not self._modelDesc:
            return

        self._name = name

        if self._modelDesc == "SimpleDLA":
            self._model = SimpleDLA(num_classes=self._numFeatures)
            self._origCkptPath = f'./checkpoint/dla_simple_{name}.pth'
        elif self._modelDesc == "ResNet34":
            self._model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34')
            self._model.fc = nn.Linear(self._model.fc.in_features, self._numFeatures)  # For resnet
            self._origCkptPath = f'./checkpoint/resnet34_{name}.pth'

        self._device = torch.device('cuda' if self._cudaAvailable else 'cpu')
        self._model = self._model.to(self._device)
        if self._cudaAvailable:
            self._model = nn.DataParallel(self._model)
            cudnn.benchmark = True

    def _set_criterion(self, desc):
        if desc:
            self._criterionDesc = desc
        elif not self._criterionDesc:
            return

        if self._criterionDesc == "CrossEntropyLoss":
            self._criterion = nn.CrossEntropyLoss()

    def _set_optimizer(self, desc: str = ""):
        if desc:
            self._optimDesc = desc
        elif not self._optimDesc:
            return

        if self._optimDesc == "SGD":
            self._optimizer = SGD(self._model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        #elif self._optimDesc == "Adam":
        #    self._optimizer = Adam()

    def _set_scheduler(self, desc: str = ""):
        if desc:
            self._schedulerDesc = desc
        elif not self._schedulerDesc:
            return

        if not self._optimizer:
            return

        if self._schedulerDesc == "SWA":
            self._scheduler = CosineAnnealingLR(self._optimizer, T_max=200)
            self._swaStart = 100
            self._swaModel = None
            self._swaScheduler = SWALR(self._optimizer, anneal_strategy="cos", anneal_epochs=20, swa_lr=0.05)

    def _set_acquirer(self, desc, size):
        if desc:
            self._acquirerDesc = desc
        elif not self._acquirerDesc:
            return

        if not self._model:
            return

        if self._acquirerDesc == "BatchBALD":
            self._acquirer = ATBatchDisagreement(size, self._model)

    def _save_model(self) -> None:
        state = {
            'weights': self._model.state_dict(),
            'optim': self._optimizer.state_dict(),
            'acc': self._bestAcc,
            'loss': self._minLoss,
        }
        torch.save(state, self._ckptPath)

    def _load_model(self) -> None:
        checkpoint = torch.load(self._ckptPath)
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

    def get_current_time(self) -> str:
        return time.strftime("%y%m%d%H%M%S", time.localtime())

    def _plot_cfm(self, mode: str, epoch: int):
        if not self._graphCFM:
            return

        # Build confusion matrix
        plt.figure(figsize=(12, 7))
        df_cm = pd.DataFrame(self._confusionMatrix, index=self._classNames, columns=self._classNames).astype(int)
        heatmap = sn.heatmap(df_cm, annot=True, fmt="d")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(f"./outputs/cfm_{self._modelDesc}_{self._name}_{mode}_{str(epoch)}_{self.get_current_time()}.png")
        plt.close()

        self._confusionMatrix = torch.zeros(self._numFeatures, self._numFeatures)

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
                    if mode == 0:
                        slide_probs = self._slideTrainProbs
                    elif mode == 1:
                        slide_probs = self._slideValProbs
                    else:
                        slide_probs = self._slideTestProbs
                    for i in range(len(prob)):
                        # print(paths[i], ": ", str(prob[i]))
                        name = os.path.normpath(paths[i]).split(os.sep)[-2]
                        if name in slide_probs:
                            slide_probs[name].append(prob[i])
                        else:
                            slide_probs[name] = [prob[i]]
                all_probs.extend(prob)
                try:
                    if self._numFeatures == 2:
                        roc_auc = roc_auc_score(all_labels, all_probs)
                    else:
                        roc_auc = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovo',
                                                labels=np.arange(self._numFeatures))
                    roc_failed = 0
                except ValueError:
                    roc_failed = 1

                if roc_failed:
                    progress_bar(index, len(loader),
                                 'ROC AUC Score: N/A (Insufficient Classes) | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                 % (loss_sum / (index + 1), 100. * correct_count / total, correct_count, total))
                else:
                    progress_bar(index, len(loader), 'ROC AUC Score: %.3f | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                 % (roc_auc, loss_sum / (index + 1), 100. * correct_count / total, correct_count,
                                    total))
                    # if roc_auc < 0.5:
                    #    print("[!] ROC AUC Score is particularly low.")
                    #    print("[-] True labels: {}".format(labels))
                    #    print("[-] Predicted probability of positive class: {}".format(prob))

                # print("[-] True labels: {}".format(labels))
                # print("[-] Predicted probability of positive class: {}".format(prob))

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
                    ax.set_title(f'Predicted: {self._classNames[predicted[i]]} Correct:{self._classNames[labels[i]]}')
                    imshow(inputs.cpu().data[i])
                    num_images_left -= 1
                if self._graphCFM:
                    for t, p in zip(labels.view(-1), predicted.view(-1)):
                        self._confusionMatrix[t.long(), p.long()] += 1

        print()
        return roc_auc, loss_sum / loss_cnt, correct_count / total

    @staticmethod
    def _print_results(mode, auroc, loss, acc):
        print(f'[*] {mode} results:')
        print('[*] ROC AUC Score: {:.4f} Loss: {:.4f} Acc: {:.4f}'.format(auroc, loss, acc))

    @staticmethod
    def _print_slide_probs(name, slide_probs):
        if not slide_probs:
            return

        print('[*] {} slide probabilities:'.format(name))
        for key in slide_probs:
            values = slide_probs[key]
            def get_avg(l):
                return sum(l) / len(l)

            def flatten(l):
                return [item for sublist in l for item in sublist]

            df = pd.DataFrame(values)
            avg = get_avg(values)
            quantile = flatten(df.quantile([0.25, 0.5, 0.95]).values.tolist())
            print('    [+] Slide name: {}'.format(key))
            print('    [+] Average Probability: {:.4f}'.format(avg))
            print("    [+] Quantiles: {:.4f}, {:.4f}, {:.4f}".format(quantile[0], quantile[1], quantile[2]))

    def _train_val_once(self, epoch: int, num_epochs: int, num_images: int = 0):
        print()
        print('[*] Epoch {}/{}'.format(epoch, num_epochs))
        swa = (epoch > self._swaStart)
        train_auroc, train_loss, train_acc = self.__train_val_once(num_images)
        self._plot_cfm("train", epoch)
        if swa:
            if not self._swaModel:
                self._swaModel = AveragedModel(self._model)
            self._swaModel.update_parameters(self._model)
            self._swaScheduler.step()
        else:
            self._scheduler.step()
        val_auroc, val_loss, val_acc = self.__train_val_once(num_images, 1)
        self._plot_cfm("validation", epoch)

        self._print_results("Training", train_auroc, train_loss, train_acc)
        self._print_results("Validation", val_auroc, val_loss, val_acc)

        self._print_slide_probs('Train', self._slideTrainProbs)
        self._print_slide_probs('Validation', self._slideValProbs)

        # Make a copy of the model if the accuracy on the validation set has improved
        if val_acc > self._bestAcc and val_loss <= self._minLoss:
            self._bestAcc = val_acc
            self._minLoss = val_loss
            self._save_model()  # Now we'll load in the best model weights
            print("[+] Model updated.")

        return [train_auroc, train_loss, train_acc * 100], [val_auroc, val_loss, val_acc * 100]

    def _plot_stats(self, epoch, train_res, val_res, append: bool = True, plot: bool = True):
        if append:
            self._trainAUROC.append(train_res[0])
            self._valAUROC.append(val_res[0])
            self._trainLoss.append(train_res[1])
            self._valLoss.append(val_res[1])
            self._trainAcc.append(train_res[2])
            self._valAcc.append(val_res[2])

        if plot:
            plt.plot(self._trainAUROC, 'g', label="Training")
            plt.plot(self._valAUROC, 'r', label="Validation")
            plt.title("ROC AUC Score Curve")
            plt.savefig(f"./outputs/rocauc_{self._modelDesc}_{self._name}_{str(epoch)}_{self.get_current_time()}.png")
            plt.close()

            plt.plot(self._trainLoss, 'g', label="Training")
            plt.plot(self._valLoss, 'r', label="Validation")
            plt.title("Loss Curve")
            plt.savefig(f"./outputs/loss_{self._modelDesc}_{self._name}_{str(epoch)}_{self.get_current_time()}.png")
            plt.close()

            plt.plot(self._trainAcc, 'g', label="Training")
            plt.plot(self._valAcc, 'r', label="Validation")
            plt.title("Accuracy Curve")
            plt.savefig(f"./outputs/acc_{self._modelDesc}_{self._name}_{str(epoch)}_{self.get_current_time()}.png")
            plt.close()

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
                self._plot_stats(epoch, train_res, val_res, True, True)

        self._reset_stats()
        if self._swaModel:
            update_bn(self._trainLoader, self._swaModel)

        run_time = time.time() - start_time
        print('[+] Training completed in {:.0f}m {:.0f}s'.format(run_time // 60, run_time % 60))

    def train_active(self, num_images: int = 0) -> None:
        total_size = len(self._trainData)
        train_size = int(0.99 * total_size)
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
                self._plot_stats(orig_len - len(pool), train_res, val_res, True, True)

        self._reset_stats()
        if self._swaModel:
            update_bn(self._trainLoader, self._swaModel)

    def test(self, num_images: int = 0):
        test_auroc, test_loss, test_acc = self.__train_val_once(num_images, 2)
        self._plot_cfm("test", 1)
        self._print_results("Test", test_auroc, test_loss, test_acc)
        self._print_slide_probs('Test', self._slideTestProbs)

        return [test_auroc, test_loss, test_acc * 100]

    def _reset_model(self) -> None:
        def _reset_weights(m):
            '''
              Try resetting model weights to avoid
              weight leakage.
            '''
            for layer in m.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
                    print(f'[-] {layer}')

        temp = self._ckptPath
        self._ckptPath = self._origCkptPath
        if os.path.isfile(self._ckptPath):
            self._load_model()
        else:
            self._model.apply(_reset_weights)
            self._set_optimizer()

        self._set_scheduler()
        self._ckptPath = temp

    def train_cross_validate(self, k_folds: int = 5, folders: list[str] = None, num_images: int = 0):
        if folders:
            total_size = len(folders)
        else:
            full_dataset = ConcatDataset([self._origTrainData, self._origTestData])
            total_size = len(full_dataset)

        fraction = 1 / k_folds
        seg = int(total_size * fraction)
        tmp_paths = []

        # tr: train, val: valid; r: right, l: left
        # [trll, trlr], [vall, valr], [trrl, trrr]
        for i in range(k_folds):
            print('[+] Fold {}'.format(i + 1))
            print('--------------------------------')
            self._bestAcc = 0.0
            self._minLoss = 100.0

            print('[*] Resetting model weights...')
            p = Path(self._origCkptPath)
            self._ckptPath = "{0}_{2}{1}".format(Path.joinpath(p.parent, p.stem), p.suffix, f"fold_{i + 1}")
            tmp_paths.append(self._ckptPath)
            self._reset_model()

            train_left_right = i * seg
            test_left = train_left_right
            test_right = i * seg + seg
            train_right_left = test_right
            train_right_right = total_size

            train_left_indices = list(range(0, train_left_right))
            test_indices = list(range(test_left, test_right))
            train_right_indices = list(range(train_right_left, train_right_right))
            train_indices = train_left_indices + train_right_indices

            if folders:
                folders_train = [folders[i] for i in train_indices]
                labels = [int("Rat_HCC_HE" in folder) for folder in folders_train]
                folders_train, folders_val, _, _ = train_test_split(folders_train, labels, test_size=0.25,
                                                                    stratify=labels)
                train_set = MedImageFolders(folders_train)
                val_set = MedImageFolders(folders_val)
                test_set = MedImageFolders([folders[i] for i in test_indices])
            else:
                train_set = torch.utils.data.dataset.Subset(full_dataset, train_indices)
                init_size = len(train_set)
                train_size = int(init_size * 0.75)
                val_size = init_size - train_size
                train_set, val_set = random_split(train_set, [train_size, val_size])
                test_set = torch.utils.data.dataset.Subset(full_dataset, test_indices)

            self._trainData = TransformedDataset(train_set, transformer=self._trainTransform)
            self._valData = TransformedDataset(val_set, transformer=self._testTransform)
            self._testData = TransformedDataset(test_set, transformer=self._testTransform)

            self._trainLoader = torch.utils.data.DataLoader(self._trainData, batch_size=self._batchSize, shuffle=True)
            self._valLoader = torch.utils.data.DataLoader(self._valData, batch_size=self._batchSize, shuffle=True)
            self._testLoader = torch.utils.data.DataLoader(self._testData, batch_size=self._batchSize, shuffle=True)
            self.train_active(num_images)
            _, loss, acc = self.test(num_images)
            acc /= 100

            if acc > self._bestAcc and loss <= self._minLoss:
                self._bestAcc = acc
                self._minLoss = loss
                self._ckptPath = self._origCkptPath
                self._save_model()
                print("[+] Model saved as best of all folds.")

        self._bestAcc = 0.0
        self._bestLoss = 100.0
        best_model = ""

        delete_paths = []
        for file in tmp_paths:
            if os.path.exists(file):
                delete_paths.append(file)
                checkpoint = torch.load(self._ckptPath)
                acc = checkpoint["acc"]
                loss = checkpoint["loss"]

                if acc > self._bestAcc and loss <= self._minLoss:
                    self._bestAcc = acc
                    self._minLoss = loss
                    best_model = file

        for file in delete_paths:
            if file == best_model:
                os.rename(best_model, self._origCkptPath)
            else:
                os.remove(file)


# https://odsc.medium.com/crash-course-pool-based-sampling-in-active-learning-cb40e30d49df

# to-do: maybe plot loss/acc change in epoch?
