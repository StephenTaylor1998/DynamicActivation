import os

from torchvision import datasets
from torchvision.transforms import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


# here is an example don't care
def classify_dataset(data_dir, not_strict=False):
    if not os.path.exists(data_dir) and not_strict:
        print("traindir ==> '%s' is not found" % data_dir)
        return
    return datasets.ImageFolder(
        data_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))


def classify_train_dataset(train_dir, not_strict=False):
    if not os.path.exists(train_dir) and not_strict:
        print("train_dir ==> '%s' is not found" % train_dir)
        return

    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # normalize,
            transforms.Resize(144),
            transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    return train_dataset


def classify_val_dataset(val_dir, not_strict=False):
    if not os.path.exists(val_dir) and not_strict:
        print("valdir ==> '%s' is not found" % val_dir)
        return

    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # normalize,
            transforms.Resize(144),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            normalize,
        ]))
    return val_dataset


def classify_test_dataset(testdir, not_strict=False):
    if not os.path.exists(testdir) and not_strict:
        print("testdir ==> '%s' is not found" % testdir)
        return

    test_dataset = datasets.ImageFolder(
        testdir,
        transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # normalize,
            transforms.Resize(144),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            normalize,
        ]))
    return test_dataset
