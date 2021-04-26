import torch


def convert_image(image):
    try:
        image_tensor = torch.from_numpy(image)
    except:
        image_tensor = image
        print("expected np.ndarray")
    return torch.unsqueeze(image_tensor, dim=0)


def model_predict(model, input_tensor):
    output = model(input_tensor)
    output = torch.squeeze(output, dim=0)
    return output


def visualization(model_prediction):
    model_prediction *= 255.0
    int64_format = model_prediction.int()
    return int64_format


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    model = torch.nn.MaxPool2d(kernel_size=1)
    # image = torch.ones((224, 224, 3), dtype=torch.float32)
    image = torch.zeros((224, 224, 3), dtype=torch.float32)
    inputs = convert_image(image)
    output = model_predict(model, inputs)
    output = visualization(output)
    plt.imshow(output)
    plt.show()

