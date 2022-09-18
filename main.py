import torchvision.datasets
from torch.optim import SGD

from ATTrainTest import *

if __name__ == '__main__':
    norm = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34')
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    acquirer = ATBatchDisagreement(64, model)

    trainer = ATTrainTest(model, criterion, optimizer, acquirer, norm, train_set, test_set,
                          128)
    trainer.set_visual_options(False, True, True)
    trainer.train(10, 0)
    trainer.test(0)
