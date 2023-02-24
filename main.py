import torchvision.datasets
from trainer import *
from med_data import MedImageFolders
from sklearn.model_selection import train_test_split
import random

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
        labels = [int("Rat_HCC_HE" in folder) for folder in folders]

        # 0: 8; 1: 20 -> 6 + 6 | 1 + 7 | 1 + 7
        # folders_train, folders_temp, labels_train, labels_temp = train_test_split(folders, labels, test_size=0.4, stratify=labels)
        # folders_val, folders_test, labels_val, labels_test = train_test_split(folders_temp, labels_temp, test_size=0.5, stratify=labels)

        folders_train = []
        labels_train = []
        folders_val = []
        labels_val = []
        folders_test = []
        labels_test = []

        remaining_0 = 6
        remaining_1 = 6
        for i in range(len(labels)):
            # Populating train folders
            if (labels[i] == 0 and remaining_0 > 0) or (labels[i] == 1 and remaining_1 > 0):
                folders_train.append(folders[i])
                labels_train.append(labels[i])
                if labels[i] == 0:
                    remaining_0 -= 1
                else:
                    remaining_1 -= 1

            # Allocating remaining 0s
            elif labels[i] == 0:
                if 0 not in labels_val:
                    folders_val.append(folders[i])
                    labels_val.append(0)
                else:
                    folders_test.append(folders[i])
                    labels_test.append(0)

            # Allocating remaining 1s
            elif labels[i] == 1:
                if labels_val.count(1) < 7:
                    folders_val.append(folders[i])
                    labels_val.append(1)
                else:
                    folders_test.append(folders[i])
                    labels_test.append(1)

        # print(folders_train)
        med_train = MedImageFolders(folders_train)
        med_val = MedImageFolders(folders_val)
        med_test = MedImageFolders(folders_test)

        trainer = ATTrainTest(med_train.classes, "ResNet34", "CrossEntropyLoss", "SGD", "BatchBALD", norm, med_train,
                              med_val, med_test, 16)
        trainer.train_cross_validate(folders=folders_train + folders_val)

    else:
        raise AssertionError("[!] Option must be 1 or 2!")

    # trainer.train(2)
    #trainer.train_active(0)
    #trainer.test()
