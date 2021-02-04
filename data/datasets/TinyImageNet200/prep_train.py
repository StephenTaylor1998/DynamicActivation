import os
import shutil

new_dataset_dir = './new_dataset_dir'


def prep(train_dir, new_dataset_dir):
    folder_names = os.listdir(train_dir)
    for folder_name in folder_names:
        folder_path = os.path.join(train_dir, folder_name)
        new_folder_path = os.path.join(new_dataset_dir, folder_name)
        os.makedirs(new_folder_path, exist_ok=True)
        image_dir = os.path.join(folder_path, 'images')
        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            dst_path = os.path.join(new_folder_path, image_name)
            print(image_path, "\t to \t", dst_path)
            shutil.copyfile(image_path, dst_path)

            # pass


# new_image_dir = os.path.join(new_dataset_dir, 'val')
# prep('val/images', new_image_dir, 'val/val_annotations.txt')

new_image_dir = os.path.join(new_dataset_dir, 'train')
prep('train', new_image_dir)
