import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim import lr_scheduler

from transfer_trainer import transfer_trainer


def active(model, acquirer, device, data, optimizer):
    train_data, pool_data, test_data = data

    test_accuracies = []
    while len(pool_data) > 0:
        print(f'Acquiring {acquirer.__class__.__name__} batch. Pool size: {len(pool_data)}')
        # get the indices of the best batch of data
        batch_indices = acquirer.select_batch(model, pool_data)
        # move that data from the pool to the training set
        move_data(batch_indices, pool_data, train_data)
        # train on it
        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=train_batch_size, pin_memory=True, shuffle=True)
        train(model, device, train_loader, optimizer, 0)

        # test the accuracy
        test_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=train_batch_size, pin_memory=True, shuffle=True)
        test_accuracies.append(test(model, device, test_loader))

    return test_accuracies


if __name__ == '__main__':
    f_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=f_transform)
    test_set, val_set = torch.utils.data.random_split(
        torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=f_transform), [5000, 5000])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=True)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    num_features = 10

    trainer = transfer_trainer(train_loader, test_loader, val_loader, model, criterion, optimizer, scheduler,
                               num_features, 'cifar.pth', 'cifar_optim.pth')

    trainer.train_test(10, cfm=True)
