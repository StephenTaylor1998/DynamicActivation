import torch
from torch.nn import Module, ReLU, Parameter


class DynamicReLU_A(Module):
    def __init__(self, inplace: bool = False, factor: float = 1.0):
        super(DynamicReLU_A, self).__init__()
        self.relu = ReLU(inplace=inplace)
        self.factor = Parameter(
            # torch.FloatTensor(1)
            torch.tensor(factor)
        )

    def forward(self, x):
        print('factor:', self.factor)
        x = self.relu(x) * self.factor
        return x


relu = DynamicReLU_A()

inputs = torch.tensor(0.1)
out = relu(inputs)
print(inputs)
print(out)
