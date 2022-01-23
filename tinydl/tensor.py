import numpy as np

from . import _tinydl


class Tensor:

    def __init__(self, data):
        if isinstance(data, _tinydl.Tensor):
            self.data = data
        else:
            self.data = _tinydl.Tensor(data)

    def __add__(self, other):
        return Tensor(_tinydl.op_add(self.data, other.data))

    def __sub__(self, other):
        return Tensor(_tinydl.op_sub(self.data, other.data))

    def __mul__(self, other):
        return Tensor(_tinydl.op_mul(self.data, other.data))

    def __truediv__(self, other):
        return Tensor(_tinydl.op_div(self.data, other.data))

    def require_grad_(self, r):
        self.data.require_grad_(r)

    def backward(self, grad):
        self.data.backward(grad.data)

    def grad(self):
        return Tensor(self.data.grad())

    def to_numpy(self):
        return self.data.to_numpy()

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
