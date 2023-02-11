import torchvision.datasets
import random
from trainer import *
from med_data import MedImageFolders


if __name__ == '__main__':
    norm = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    option = 2

    # CIFAR
    if option == 1:
        cifar_train = torchvision.datasets.CIFAR10(root='../data', train=True, download=True)
        cifar_test = torchvision.datasets.CIFAR10(root='../data', train=False, download=True)
        trainer = ATTrainTest(cifar_train.classes, "SimpleDLA", "CrossEntropyLoss", "SGD", "BatchBALD", norm,
                              cifar_train, None, cifar_test, resume=True)

    # Medical
    elif option == 2:
        ROOT_DIR = "F:\\Processed"
        folders = []
        for x in os.walk(ROOT_DIR):
            if x[0].count("\\") > 2:
                folders.append(x[0])
        random.shuffle(folders)

        # Hardcoded for now
        train_folders = folders[:22]
        val_folders = folders[22:25]
        test_folders = folders[25:28]

        med_train = MedImageFolders(train_folders)
        med_val = MedImageFolders(val_folders)
        med_test = MedImageFolders(test_folders)

        trainer = ATTrainTest(med_train.classes, "ResNet34", "CrossEntropyLoss", "SGD", "BatchBALD", norm, med_train,
                              med_val, med_test, 16)

    else:
        raise AssertionError("[!] Option must be 1 or 2!")

    trainer.set_visual_options(False, True, True)
    # trainer.train(10)
    # trainer.train_crossval()
    trainer.train_active(0)
    trainer.test()
