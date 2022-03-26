import numpy as np

from . import _tinydl
from . import operators as opr


class Tensor:

    def __init__(self, data):
        if isinstance(data, _tinydl.Tensor):
            self.data = data
        elif isinstance(data, np.ndarray):
            self.data = _tinydl.Tensor(data)
        else:
            self.data = _tinydl.Tensor(np.array(data, dtype=np.float32))

    def __add__(self, other):
        return opr.add(self, other)

    def __sub__(self, other):
        return opr.sub(self, other)

    def __mul__(self, other):
        return opr.mul(self, other)

    def __truediv__(self, other):
        return opr.div(self, other)

    def sum(self, axis, keep_dim=False):
        return opr.sum(self, axis, keep_dim)

    def mean(self, axis, keep_dim=False):
        return opr.mean(self, axis, keep_dim)

    def log(self):
        return opr.log(self)

    def exp(self):
        return opr.exp(self)

    def relu(self):
        return opr.relu(self)

    def view(self, *axis):
        return opr.view(self, axis)

    def requires_grad_(self, r):
        self.data.requires_grad_(r)

    @property
    def requires_grad(self):
        return self.data.requires_grad

    def zero_grad(self):
        return self.data.zero_grad()

    def backward(self, grad):
        self.data.backward(grad.data)

    def grad(self):
        return Tensor(self.data.grad())

    def to_numpy(self):
        return self.data.to_numpy()
    numpy = to_numpy

    def shape(self):
        return self.numpy().shape

    def __str__(self):
        return self.numpy().__str__()

    def __repr__(self):
        return "Tensor({})".format(self.numpy().__str__())

    def item(self):
        pass

    def __len__(self):
        pass
# class Tensor(_tinydl.Tensor):
#     # def __new__(cls, data, *args, **kwargs):
#     #     print(cls)
#     #     if isinstance(data, _tinydl.Tensor):
#     #         print("+++++++")
#     #         return data
#     #     else:
#     #         return _tinydl.Tensor.__new__(cls, data, *args, **kwargs)
# 
#     def __init__(self, data):
#         super().__init__(data)
# 
#     def __add__(self, other):
#         return Tensor(_tinydl.op_add(self, other))
# 
#     def __sub__(self, other):
#         return Tensor(_tinydl.op_sub(self, other))
# 
#     def __mul__(self, other):
#         return Tensor(_tinydl.op_mul(self, other))
# 
#     def __truediv__(self, other):
#         return Tensor(_tinydl.op_div(self, other))
