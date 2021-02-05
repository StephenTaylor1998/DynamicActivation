import os
import shutil


def copy_weights(arg):
    model_name = arg.arch
    epoch = arg.epochs
    batch_size = arg.batch_size
    learning_rate = arg.lr
    datasets_dir = arg.data.strip()
    datasets = datasets_dir.split('/')[-2]

    folder_name = '%s_epoch%d_bs%d_lr%.1e_%s' % \
                  (model_name, epoch, batch_size, learning_rate, datasets)
    # print(folder_name)
    folder_path = os.path.join('./data/weights', folder_name)
    print('making dir ', folder_path)
    os.makedirs(folder_path, exist_ok=True)

    checkpoint_name = 'checkpoint.pth.tar'
    model_best_name = 'model_best.pth.tar'

    print("copy file from %s to %s" % (
        os.path.join('./data', checkpoint_name),
        os.path.join(folder_path, checkpoint_name)))
    shutil.copyfile(os.path.join('./data', checkpoint_name),
                    os.path.join(folder_path, checkpoint_name))

    print("copy file from %s to %s" % (
        os.path.join('./data', model_best_name),
        os.path.join(folder_path, model_best_name)))
    shutil.copyfile(os.path.join('./data', model_best_name),
                    os.path.join(folder_path, model_best_name))
