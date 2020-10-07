from efficientnet_pytorch import EfficientNet

__all__ = ['down_load_weight', 'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']


def down_load_weight():
    _b0 = b0()
    _b1 = b1()
    _b2 = EfficientNet.from_pretrained('efficientnet-b2')
    _b3 = EfficientNet.from_pretrained('efficientnet-b3')
    _b4 = EfficientNet.from_pretrained('efficientnet-b4')
    _b5 = EfficientNet.from_pretrained('efficientnet-b5')
    _b6 = EfficientNet.from_pretrained('efficientnet-b6')
    _b7 = EfficientNet.from_pretrained('efficientnet-b7')
    return [_b0, _b1, _b2, _b3, _b4, _b5, _b6, _b7]


def b0():
    return EfficientNet.from_pretrained('efficientnet-b0')

def b1():
    return EfficientNet.from_pretrained('efficientnet-b1')

def b2():
    return EfficientNet.from_pretrained('efficientnet-b2')

def b3():
    return EfficientNet.from_pretrained('efficientnet-b3')

def b4():
    return EfficientNet.from_pretrained('efficientnet-b4')

def b5():
    return EfficientNet.from_pretrained('efficientnet-b5')

def b6():
    return EfficientNet.from_pretrained('efficientnet-b6')

def b7():
    return EfficientNet.from_pretrained('efficientnet-b7')

if __name__ == '__main__':
    print(__all__)
    down_load_weight()
