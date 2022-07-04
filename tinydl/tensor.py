import numpy as np

from . import _tinydl
from ._tinydl import DataType
from . import operators as opr


def as_numpy_dtype(dtype):
    if dtype is None:
        return None
    mapping = {
        DataType.float32: np.float32,
        DataType.uint64: np.uint64,
        DataType.bool: np.bool,
    }
    if isinstance(dtype, DataType):
        return mapping[dtype]
    assert dtype in set(mapping.values())
    return dtype


class Tensor:

    def __init__(self, data, dtype=None):
        if isinstance(data, _tinydl.Tensor):
            self.data = data
        elif isinstance(data, np.ndarray):
            self.data = _tinydl.Tensor(data)
        else:
            arr = np.array(data, dtype=as_numpy_dtype(dtype))
            if arr.dtype in [np.float32, np.float64]:
                self.data = _tinydl.Tensor(arr.astype(np.float32))
            elif arr.dtype in [np.uint64]:
                self.data = _tinydl.Tensor(arr.astype(np.uint64))
            elif arr.dtype in [np.bool]:
                self.data = _tinydl.Tensor(arr.astype(np.bool))
            else:
                raise NotImplementedError


    def __add__(self, other):
        return opr.add(self, other)

    def __sub__(self, other):
        return opr.sub(self, other)

    def __mul__(self, other):
        return opr.mul(self, other)

    def __truediv__(self, other):
        return opr.div(self, other)

    def __eq__(self, other):
        return opr.equal(self, other)

    def __lt__(self, other):
        return opr.less_then(self, other)

    def __le__(self, other):
        return opr.less_equal(self, other)

    def __gt__(self, other):
        return opr.greater_then(self, other)

    def __ge__(self, other):
        return opr.greater_equal(self, other)

    def sum(self, axis, keep_dim=False):
        return opr.sum(self, axis, keep_dim)

    def mean(self, axis, keep_dim=False):
        return opr.mean(self, axis, keep_dim)

    def min(self, axis, keep_dim=False):
        return opr.min(self, axis, keep_dim)

    def max(self, axis, keep_dim=False):
        return opr.max(self, axis, keep_dim)

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

    def backward(self, grad=None):
        if grad is None:
            grad = Tensor([1.])
        self.data.backward(grad.data)

    def grad(self):
        return Tensor(self.data.grad())

    def to_numpy(self):
        if self.dtype() == _tinydl.float32:
            return self.data._to_numpy_float()
        elif self.dtype() == _tinydl.uint64:
            return self.data._to_numpy_uint64()
        elif self.dtype() == _tinydl.bool:
            return self.data._to_numpy_bool()
    numpy = to_numpy

    def shape(self):
        return self.numpy().shape

    def __str__(self):
        return self.numpy().__str__()

    def __repr__(self):
        return "Tensor({})".format(self.numpy().__str__())

    def dtype(self):
        return self.data.dtype()

    def item(self):
        pass

    def as_float(self):
        return opr.as_float32(self)

    def as_bool(self):
        return opr.as_bool(self)

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
