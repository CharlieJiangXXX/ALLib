import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader  # Somehow this has to be included to resolve a warning


# @class ATBase
# @abstract Base class for all trainers/testers in the Active Trainer Module.
# @discussion

class ATBase:
    def __init__(self, train_loader: DataLoader, test_loader: DataLoader, model: torchvision.models,
                 criterion: nn.modules.loss, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler,
                 num_features: int, weight_path: str, optim_path: str,
                 best_acc: int = 0.0, min_loss: int = 0.0) -> None:

        # Determine if file paths are valid
        if not os.path.isfile(weight_path) or not os.path.isfile(optim_path):
            raise FileNotFoundError
        self.weight_path = weight_path
        self.optim_path = optim_path

        # Store dataloaders
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Device & model initialization
        self.model = model
        self.model.fc = nn.Linear(model.fc.in_features, num_features)  # Number of features in fully connected layer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        if self.device == torch.device('cuda'):
            self.model = torch.nn.DataParallel(self.model)
            cudnn.benchmark = True

        # Criterion, optimizer, scheduler
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Load the imported model & store the best accuracy
        self.best_acc = best_acc
        self.min_loss = min_loss
        self._load_model()

        self.e_loss = []
        self.e_acc = []
        self.temp_loss = 0.0
        self.temp_corrects = 0
        self.cfm = torch.zeros(num_features, num_features)

    def _save_model(self):
        torch.save(self.model.state_dict(), self.weight_path)
        torch.save(self.optimizer.state_dict(), self.optim_path)

    def _load_model(self):
        self.model.load_state_dict(torch.load(self.weight_path))
        self.optimizer.load_state_dict(torch.load(self.optim_path))
