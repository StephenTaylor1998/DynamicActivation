from efficientnet_pytorch import EfficientNet

__all__ = ['__down_load_weight__', 'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']


def __down_load_weight__():
    _b0 = b0()
    _b1 = b1()
    _b2 = b2()
    _b3 = b3()
    _b4 = b4()
    _b5 = b5()
    _b6 = b6()
    _b7 = b7()
    return [_b0, _b1, _b2, _b3, _b4, _b5, _b6, _b7]


def b0(pretrained=True, num_classes=1000):
    if pretrained:
        return EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
    else:
        return EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)


def b1(pretrained=True, num_classes=1000):
    if pretrained:
        return EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)
    else:
        return EfficientNet.from_name('efficientnet-b1', num_classes=num_classes)


def b2(pretrained=True, num_classes=1000):
    if pretrained:
        return EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)
    else:
        return EfficientNet.from_name('efficientnet-b2', num_classes=num_classes)


def b3(pretrained=True, num_classes=1000):
    if pretrained:
        return EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
    else:
        return EfficientNet.from_name('efficientnet-b3', num_classes=num_classes)


def b4(pretrained=True, num_classes=1000):
    if pretrained:
        return EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
    else:
        return EfficientNet.from_name('efficientnet-b4', num_classes=num_classes)


def b5(pretrained=True, num_classes=1000):
    if pretrained:
        return EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes)
    else:
        return EfficientNet.from_name('efficientnet-b5', num_classes=num_classes)


def b6(pretrained=True, num_classes=1000):
    if pretrained:
        return EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes)
    else:
        return EfficientNet.from_name('efficientnet-b6', num_classes=num_classes)


def b7(pretrained=True, num_classes=1000):
    if pretrained:
        return EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)
    else:
        return EfficientNet.from_name('efficientnet-b7', num_classes=num_classes)


if __name__ == '__main__':
    print(__all__)
    _down_load_weight()
