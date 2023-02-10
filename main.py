import torchvision.datasets
from transformed_dataset import TransformedDataset
from trainer import *

if __name__ == '__main__':
    norm = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # CIFAR
    cifar_train = torchvision.datasets.CIFAR10(root='../data', train=True, download=True)
    cifar_test = torchvision.datasets.CIFAR10(root='../data', train=False, download=True)

    # Medical
    # medical = torchvision.datasets.ImageFolder(root="/media/cjiang/Extreme SSD/Processed/")
    # medical_size = len(medical)
    # medical_train_size = int(0.8 * medical_size)
    # medical_test_size = medical_size - medical_train_size
    # medical_train, medical_test = random_split(medical, [medical_train_size, medical_test_size])

    trainer = ATTrainTest(cifar_train.classes, "SimpleDLA", "CrossEntropyLoss", "SGD", "BatchBALD", norm, cifar_train, cifar_test, 128)
    # trainer = ATTrainTest(medical.classes, "ResNet34", "CrossEntropyLoss", "SGD", "BatchBALD", norm, medical_train, medical_test, 16)
    trainer.set_visual_options(False, True, True)
    # trainer.train(10)
    # trainer.train_crossval()
    trainer.train_active(0)
    trainer.test()
