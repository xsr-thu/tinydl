from .. import Tensor
from collections import OrderedDict


class SGD:
    def __init__(self, lr):
        self.lr = Tensor([lr])
        self.parameters = None

    def attach(self, named_parameters):
        self.named_parameters = OrderedDict(named_parameters)

    def step(self):
        for n, p in self.named_parameters.items():
            p.set_value(p - self.lr * p.grad())

    def zero_grad(self):
        for p in self.named_parameters.values():
            p.zero_grad()
