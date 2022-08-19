import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision

from ATAcquisition.ATBatchDisagreement import *


def imshow(inp, title: str = None) -> None:
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# @class ATTrainTest
# @abstract
# @discussion

class ATTrainTest:
    def __init__(self, model: torchvision.models, criterion: nn.modules.loss, optimizer: optim.Optimizer,
                 scheduler: optim.lr_scheduler, acquirer: ATBatchDisagreement, num_features: int,
                 train_data: torchvision.datasets, test_data: torchvision.datasets = None,
                 val_data: torchvision.datasets = None, batch_size: int = 128,
                 acc_path: str = 'at_acc_loss.txt', weight_path: str = 'at_weights.pth',
                 optim_path: str = 'at_optim.pth') -> None:

        self._weightPath = weight_path
        self._optimPath = optim_path

        # Device & model initialization
        self._model = model
        self._model.fc = nn.Linear(model.fc.in_features, num_features)  # Number of features in fully connected layer
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = self._model.to(self._device)
        if self._device == torch.device('cuda'):
            self._model = torch.nn.DataParallel(self._model)
            cudnn.benchmark = True

        # Criterion, optimizer, scheduler
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._acquirer = acquirer

        # Dataloader initialization
        if batch_size > 0:
            self._batchSize = batch_size
        else:
            self._batchSize = 128

        if not train_data:
            raise AssertionError
        self._origTrainData = train_data
        self.set_train_data(train_data)
        if test_data:
            self.set_test_data(test_data)
        if val_data:
            self.set_val_data(val_data)

        # Store the best accuracy & load the imported model if it exists
        self._bestAcc = 0.0
        self._minLoss = 100.0
        self._accPath = acc_path
        if os.path.isfile(acc_path) and os.path.isfile(weight_path) and os.path.isfile(optim_path):
            self._load_model()

        self._confusionMatrix = torch.zeros(num_features, num_features)

    def __del__(self) -> None:
        self.get_best_acc()
        self.get_min_loss()

    def _save_model(self) -> None:
        with open(self._accPath, 'w') as f:
            f.write("Best accuracy: {:.4f}\n".format(self._bestAcc))
            f.write("Minimum loss: {:.4f}".format(self._minLoss))
        torch.save(self._model.state_dict(), self._weightPath)
        torch.save(self._optimizer.state_dict(), self._optimPath)

    def _load_model(self) -> None:
        try:
            with open(self._accPath, 'r') as f:
                acc = f.readline()
                self._bestAcc = float(acc.replace("Best accuracy: ", "").strip())
                loss = f.readline()
                self._minLoss = float(loss.replace("Minimum loss: ", "").strip())
        except ValueError:
            pass

        self._model.load_state_dict(torch.load(self._weightPath))
        self._optimizer.load_state_dict(torch.load(self._optimPath))

    def set_train_data(self, dataset: torchvision.datasets) -> None:
        self._trainData = dataset
        self._trainLoader = DataLoader(dataset, batch_size=self._batchSize, shuffle=True)

    def set_test_data(self, dataset: torchvision.datasets) -> None:
        self._testData = dataset
        self._classNames = dataset.classes
        self._testLoader = DataLoader(dataset, batch_size=self._batchSize, shuffle=True)

    def set_val_data(self, dataset: torchvision.datasets) -> None:
        self._valData = dataset
        self._valLoader = DataLoader(dataset, batch_size=self._batchSize, shuffle=True)

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

        # ROC-AUC
        # tn, fp, fn, tp = self.cfm.ravel()
        # tpr = tp / (tp + fn)
        # fpr = fp / (fp + tn)
        # roc_values.append([tpr, fpr])
        # tpr_values, fpr_values = zip(*roc_values)
        # fig, ax = plt.subplots(figsize=(10,7))
        # ax.plot(fpr_values, tpr_values)
        # ax.plot(np.linspace(0, 1, 100),
        #        np.linspace(0, 1, 100),
        #        label='baseline',
        #        linestyle='--')
        # plt.title('Receiver Operating Characteristic Curve', fontsize=18)
        # plt.ylabel('TPR', fontsize=16)
        # plt.xlabel('FPR', fontsize=16)
        # plt.legend(fontsize=12);

    def _train_test_once(self, train: bool = True, prob: bool = False, cfm: bool = False, img: int = 0) -> (float, int):
        # First set to eval mode
        self._model.train() if train else self._model.eval()

        orig_img = img
        temp_loss = 0.0
        temp_corrects = 0

        if train and not self._trainLoader:
            print("[!] self._trainLoader not allocated. Please first invoke set_train_data().")
            return -1.0, -1
        elif not train and not self._testLoader:
            print("[!] self._testLoader not allocated. Please first invoke set_test_data().")
            return -1.0, -1

        for inputs, labels in (self._trainLoader if train else self._testLoader):
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)
            self._optimizer.zero_grad()

            # Enable gradient for training
            with torch.set_grad_enabled(train):
                outputs = self._model(inputs)
                loss = self._criterion(outputs, labels)
                predicted = outputs.argmax(dim=1, keepdim=True)
                if train:
                    loss.backward()
                    self._optimizer.step()

            for i in range(inputs.size()[0]):
                if img == 0:
                    break
                img -= 1
                ax = plt.subplot(orig_img // 2, 2, img)
                ax.axis('off')
                ax.set_title(f'predicted: {self._classNames[predicted[i]]}')
                imshow(inputs.cpu().data[i])

            temp_loss += loss.item() * inputs.size(0)
            temp_corrects += predicted.eq(labels.view_as(predicted)).sum().item()
            if prob:
                m = nn.Softmax(dim=1)
                print(m(outputs))
            if cfm:
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    self._confusionMatrix[t.long(), p.long()] += 1

        if train:
            self._scheduler.step()
        return temp_loss, temp_corrects

    def _train_test(self, mode: int, num_epochs: int = 10, num_images: int = 0,
                    prob: bool = False, cfm: bool = False, loss: bool = True) -> (list, list):
        start_time = time.time()
        self._bestAcc = 0.0
        temp_loss = 0.0
        temp_corrects = 0
        epoch_loss = []
        epoch_acc = []
        mode_str = "Training & Testing"

        if mode == 0:
            dataset_size = len(self._trainLoader) * self._trainLoader.batch_size
            mode_str = "Training"
            self.set_train_data(self._acquirer.select_batch((self._origTrainData, self._testData)))
        else:
            dataset_size = len(self._testLoader) * self._testLoader.batch_size
            if mode == 1:
                mode_str = "Testing"

        for epoch in range(1, num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            if mode == 0 or mode == 2:
                temp_loss, temp_corrects = self._train_test_once()
            if mode == 1 or mode == 2:
                temp_loss, temp_corrects = self._train_test_once(False, prob, cfm, num_images)

            epoch_loss.append(temp_loss / dataset_size)
            epoch_acc.append(temp_corrects / dataset_size)
            print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss[-1], epoch_acc[-1]))
            if cfm:
                self._plot_cfm()

            # Make a copy of the model if the accuracy on the validation set has improved
            if epoch_acc[-1] > self._bestAcc and epoch_loss[-1] <= self._minLoss:
                print("OK")
                self._bestAcc = epoch_acc[-1]
                self._minLoss = epoch_loss[-1]
                self._save_model()  # Now we'll load in the best model weights

            print()

        if loss:
            plt.plot(epoch_loss)
            plt.plot(epoch_acc)
        run_time = time.time() - start_time
        print('[+] {} completed in {:.0f}m {:.0f}s'.format(mode_str, run_time // 60, run_time % 60))
        self.get_best_acc()
        return epoch_loss, epoch_acc

    def train(self, num_epochs: int = 10) -> None:
        self._train_test(0, num_epochs)

    def test(self, num_epochs: int = 10, prob: bool = False, cfm: bool = False, loss: bool = True) -> None:
        self._train_test(1, num_epochs, prob, cfm, loss)

    def train_test(self, num_epochs: int = 10, prob: bool = False, cfm: bool = False, loss: bool = True) -> None:
        self._train_test(2, num_epochs, prob, cfm, loss)
