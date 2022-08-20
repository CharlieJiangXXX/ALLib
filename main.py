from torch.optim import lr_scheduler
from torchvision import transforms

from ATTrainTest import *

if __name__ == '__main__':
    f_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=f_transform)
    test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=f_transform)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    acquirer = ATBatchDisagreement(64, model)
    num_features = 10

    train_test = ATTrainTest(model, criterion, optimizer, scheduler, acquirer, num_features, train_set, test_set,
                             None, 128)
    train_test.train_test(100)
