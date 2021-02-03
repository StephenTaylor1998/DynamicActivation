import os
import numpy as np
import torch
# import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from PIL import Image
from torchvision import transforms
from core import models


def create_model_and_load_weight(
        model_name='b0',
        resume='./data/checkpoint.pth.tar',
        gpu=1,
        num_classes=1000):
    if gpu is not None:
        print("Use GPU: {} for evaluating".format(gpu))

    # create model
    print("=> using pre-trained model '{}'".format(model_name))

    # < Attention >
    # (1)if you want to use models in core/models or torchvision.models use this.
    # you can also register your model in core/models/__inti__.py then use this.
    model = models.__dict__[model_name](pretrained=False, num_classes=num_classes)

    # (2)if you want to use models you defined, or import from other projects.
    from efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_name('efficientnet-b0', num_classes=10)
    # < /Attention >

    checkpoint = torch.load(resume)
    print('use parameters in epoch:', checkpoint['epoch'])
    # if you would like to use cpu, remove model.cuda(gpu)
    try:
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda(gpu)
    except:
        print('load distributed paramrters')
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda(gpu)

    # cudnn.benchmark = True
    # model.eval()
    print('parameters loaded!')
    return model


transform = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_image(image_path='./demo01.jpg'):
    img = Image.open(image_path)
    img = transform(img)
    image_array = np.array(img)
    print('origin image shape ===>', image_array.shape)
    torch_array = torch.from_numpy(image_array).unsqueeze(0)
    torch_array = torch_array.type(torch.float32)
    print('add the dim of batch ===>', torch_array.shape)
    return torch_array


def pred_in_folder(path, model):
    model.eval()
    image_list = os.listdir(path)
    for image_name in image_list:
        image_path = os.path.join(path, image_name)
        image = load_image(image_path)
        output = model(image)
        print(output)
        print(output.argmax())
    pass


if __name__ == '__main__':
    # < train model or dowmload weights/chechpoint before run this demo > #
    image = load_image(image_path='./demo01.jpg')
    model = create_model_and_load_weight(
        model_name='b0',
        # resume='./data/model_best.pth.tar',
        resume='./data/checkpoint.pth.tar',
        gpu=0,
        num_classes=10)

    model.eval()
    with torch.no_grad():
        # predict single image
        output = model(image)
        print(output)
        print(output.argmax())
        # predict images in folder
        pred_in_folder('./data/demo_data/native/09', model)
