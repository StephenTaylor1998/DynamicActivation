import os
from torchvision import datasets
from torchvision.transforms import transforms
from .custom_transform import *

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def classify_dataset(data_dir, transform, not_strict=False):
    if not os.path.exists(data_dir) and not_strict:
        print("path ==> '%s' is not found" % data_dir)
        return

    return datasets.ImageFolder(data_dir, transform)


# train dataset example
def classify_train_dataset(train_dir, transform=ImageNetTrainTransform):
    return datasets.ImageFolder(train_dir, transform)


# val dataset example
def classify_val_dataset(val_dir, transform=ImageNetValidationTransform):
    return datasets.ImageFolder(val_dir, transform)


# test dataset example
def classify_test_dataset(testdir, transform=ImageNetTestTransform):
    return datasets.ImageFolder(testdir, transform)
