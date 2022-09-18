import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torch.backends.cudnn as cudnn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import *
from torch.optim.swa_utils import *
from torchvision import transforms

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


CHECKPOINT_PATH = './checkpoint/ckpt.pth'


# @class ATTrainTest
# @abstract
# @discussion

class ATTrainTest:
    def __init__(self, model: nn.Module, criterion: nn.modules.loss,
                 optimizer: Optimizer, acquirer: ATBatchDisagreement,
                 norm: transforms.Normalize, train_data: Dataset, test_data: Dataset = None,
                 batch_size: int = 512, resume: bool = True) -> None:

        # Dataloader initialization
        self._batchSize = 128
        if batch_size > 0:
            self._batchSize = batch_size

        if not train_data:
            raise AssertionError

        self._norm = norm
        self.set_train_data(train_data)
        if test_data:
            self.set_test_data(test_data)

        # Device & model initialization
        self._model = model
        self._model.fc = nn.Linear(model.fc.in_features, self._numFeatures)
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = self._model.to(self._device)
        if torch.cuda.is_available():
            self._model = nn.DataParallel(self._model)
            cudnn.benchmark = True

        # Criterion, optimizer, scheduler
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = CosineAnnealingLR(self._optimizer, T_max=200)
        self._swaModel = None
        self._swaScheduler = SWALR(self._optimizer, anneal_strategy="cos", anneal_epochs=20, swa_lr=0.05)
        self._acquirer = acquirer

        # Store the best accuracy & load the imported model if it exists
        self._bestAcc = 0.0
        self._minLoss = 100.0

        if resume:
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if os.path.isfile(CHECKPOINT_PATH):
                self._load_model()

        self._printProb = False
        self._graphCFM = False
        self._graphLoss = False
        self._confusionMatrix = torch.zeros(self._numFeatures, self._numFeatures)

    def __del__(self) -> None:
        self.get_best_acc()
        self.get_min_loss()

    def _save_model(self) -> None:
        state = {
            'weights': self._model.state_dict(),
            'optim': self._optimizer.state_dict(),
            'acc': self._bestAcc,
            'loss': self._minLoss,
        }
        torch.save(state, CHECKPOINT_PATH)

    def _load_model(self) -> None:
        checkpoint = torch.load(CHECKPOINT_PATH)
        self._model.load_state_dict(checkpoint['weights'])
        self._optimizer.load_state_dict(checkpoint['optim'])
        self._bestAcc = checkpoint['acc']
        self._minLoss = checkpoint['loss']

    def set_train_data(self, train: Dataset) -> None:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self._norm,
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            self._norm,
        ])

        train, val = random_split(train, (int(0.9 * len(train)), int(0.1 * len(train))))
        train.dataset.transform = train_transform
        val.dataset.transform = val_transform

        self._trainLoader = DataLoader(train, batch_size=self._batchSize,
                                       pin_memory=torch.cuda.is_available(), num_workers=2)
        self._valLoader = DataLoader(val, batch_size=self._batchSize,
                                     pin_memory=torch.cuda.is_available(), num_workers=2)

    def set_test_data(self, test: Dataset) -> None:
        self._classNames = test.classes
        self._numFeatures = len(test.classes)

        test = Subset(test, indices=torch.arange(len(test)))
        transform = transforms.Compose([
            transforms.ToTensor(),
            self._norm,
        ])
        test.dataset.transform = transform
        self._testLoader = DataLoader(test, batch_size=self._batchSize,
                                      pin_memory=torch.cuda.is_available(), shuffle=False, num_workers=2)

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

    def _train_test_once(self, num_images: int = 0, mode: int = 0) -> (float, int):
        # First set to eval mode
        if mode == 0:
            print("train")
            self._model.train()
            loader = self._trainLoader
        else:
            print("eval")
            self._model.eval()
            loader = self._valLoader if mode == 1 else self._testLoader
        assert loader, "[!] Dataloader is not allocated."

        num_images_left = num_images
        loss_sum = 0.0
        loss = 0.0
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
                loss_sum += loss.item() * inputs.size(0)
                predicted = outputs.argmax(dim=1, keepdim=True)
                total += labels.size(0)
                correct_count += predicted.eq(labels.view_as(predicted)).sum().item()
                loss = loss_sum / (index + 1)
                print("Loss: %.3f | Acc: %.3f%% (%d/%d)" % (loss, 100. * correct_count / total, correct_count, total))

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

        return loss, correct_count / total

    def train(self, num_epochs: int = 10, num_images: int = 0) -> (list, list):
        start_time = time.time()
        epoch_loss = []
        epoch_acc = []

        # self.set_train_data(self._acquirer.select_batch(self._testLoader.dataset))

        for epoch in range(1, num_epochs + 1):
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
            print()

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
