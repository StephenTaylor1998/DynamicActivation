import os
import shutil

new_dataset_dir = './new_dataset_dir'


def prep(origin_image_dir, new_image_dir, annotation_file):
    with open(annotation_file) as f:
        lines = f.readlines()
        print(lines)
        for line in lines:
            # print(line.split('\t'))
            items = line.split('\t')
            # process val
            image_name = items[0]
            folder_name = items[1]

            _origin_image_path = os.path.join(origin_image_dir, image_name)
            _new_image_dir = os.path.join(new_image_dir, folder_name)
            os.makedirs(_new_image_dir, exist_ok=True)
            new_image_path = os.path.join(_new_image_dir, image_name)
            shutil.copyfile(_origin_image_path, new_image_path)
            print(_origin_image_path, '\t to \t', new_image_path)


new_image_dir = os.path.join(new_dataset_dir, 'val')
prep('val/images', new_image_dir, 'val/val_annotations.txt')
