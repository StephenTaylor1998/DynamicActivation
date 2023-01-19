from . import data


def get_data_by_name(name: str, **kwargs):
    if name.islower() and not name.startswith("_"):
        data_file = data.__dict__[name]
        train_dataset = data_file.train_dataset(**kwargs)
        val_dataset = data_file.val_dataset(**kwargs)
        test_dataset = data_file.test_dataset(**kwargs)
    else:
        print(f"[ERROR] Data name you selected '{name}' is not support, but can be registered.")
        print("[WARNING] Custom dataset should be added into 'core/dataset/data/*', "
              "and import in file 'core/dataset/data/__init__.py'.")
        raise NameError

    return train_dataset, val_dataset, test_dataset
