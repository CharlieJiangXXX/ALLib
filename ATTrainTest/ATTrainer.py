from ATTrainTest.ATTester import *


# @class ATTester
# @abstract
# @discussion

class ATTrainer(ATTester):
    def _train_once(self):
        self.model.train()
        self.temp_loss = 0.0
        self.temp_corrects = 0

        for index, sample in enumerate(self.train_loader):
            inputs = sample[0].to(self.device)
            labels = sample[1].to(self.device)
            self.optimizer.zero_grad()

            # Enable gradient for training
            torch.set_grad_enabled(True)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()

    def train(self, num_epochs: int = 10):
        start_time = time.time()
        self.best_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            self._train_once()
            self._process_epoch_stats()
            print()

        run_time = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(run_time // 60, run_time % 60))
        print('Best Accuracy: {:4f}'.format(self.best_acc))
