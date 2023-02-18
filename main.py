import torchvision.datasets
from trainer import *
from med_data import MedImageFolders
from sklearn.model_selection import train_test_split

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
        labels = [int("Rat_HCC_HE" in folder) for folder in folders]

        folders_train, folders_temp, labels_train, labels_temp = train_test_split(folders, labels, test_size=0.4,
                                                                                  random_state=1, stratify=labels)
        folders_val, folders_test, labels_val, labels_test = train_test_split(folders_temp, labels_temp, test_size=0.5,
                                                                              random_state=1, stratify=labels_temp)

        print(labels_train, labels_val, labels_test)
        med_train = MedImageFolders(folders_train)
        med_val = MedImageFolders(folders_val)
        med_test = MedImageFolders(folders_test)

        trainer = ATTrainTest(med_train.classes, "ResNet34", "CrossEntropyLoss", "SGD", "BatchBALD", norm, med_train,
                              med_val, med_test, 16)

    else:
        raise AssertionError("[!] Option must be 1 or 2!")

    trainer.set_visual_options(False, True, True)
    # trainer.train(2)
    # trainer.train_crossval()
    trainer.train_active(0)
    trainer.test()
