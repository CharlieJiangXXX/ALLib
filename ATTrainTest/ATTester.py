from ATTrainTest.ATBase import *


# @class ATTester
# @abstract Base class for all testers in the Active Trainer module.
# @discussion

class ATTester(ATBase):
    def _test_once(self, prob: bool = False, cfm: bool = False):
        self.model.eval()
        self.temp_loss = 0.0
        self.temp_corrects = 0

        for index, sample in enumerate(self.test_loader):
            inputs = sample[0].to(self.device)
            labels = sample[1].to(self.device)
            self.optimizer.zero_grad()

            torch.set_grad_enabled(False)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            self.temp_loss += loss.item() * inputs.size(0)
            self.temp_corrects += torch.sum(preds == labels.data)
            if prob:
                m = nn.Softmax(dim=1)
                print(m(outputs))
            if cfm:
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    self.cfm[t.long(), p.long()] += 1

        self.scheduler.step()

    def _process_epoch_stats(self):
        self.e_loss.append(self.temp_loss / (len(self.test_loader) * self.test_loader.batch_size))
        self.e_acc.append(self.temp_corrects.cpu().double() / (len(self.test_loader) * self.test_loader.batch_size))
        print('Loss: {:.4f} Acc: {:.4f}'.format(self.e_loss[-1], self.e_acc[-1]))
        # Make a copy of the model if the accuracy on the validation set has improved

        # Should we add a check for loss here?
        if self.e_acc[-1] > self.best_acc:
            if self.e_loss[-1] <= self.min_loss:
                self.best_acc = self.e_acc[-1]
                self.min_loss = self.e_loss[-1]
                self._save_model()  # Now we'll load in the best model weights

    def _plot_cfm(self):
        # Constant for classes
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'sheep', 'truck']

        # Build confusion matrix
        plt.figure(figsize=(12, 7))
        df_cm = pd.DataFrame(self.cfm, index=class_names, columns=class_names).astype(int)
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

    def test(self, num_epochs: int = 10, prob: bool = False, cfm: bool = False, loss: bool = True):
        start_time = time.time()
        self.best_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            self._test_once(prob, cfm)
            self._process_epoch_stats()
            if cfm:
                self._plot_cfm()
            print()

        if loss:
            plt.plot(self.e_loss)
            plt.plot(self.e_acc)
        run_time = time.time() - start_time
        print('Testing complete in {:.0f}m {:.0f}s'.format(run_time // 60, run_time % 60))
        print('Best Accuracy: {:4f}'.format(self.best_acc))
