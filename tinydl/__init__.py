from .tensor import Tensor
from ._tinydl import op_add as add
from ._tinydl import op_sub as sub
from ._tinydl import op_mul as div
from ._tinydl import op_relu as relu
from ._tinydl import op_reduce_sum as sum
from ._tinydl import op_matmul
from . import _tinydl



def matmul(x, y):
    return Tensor(op_matmul(x.data, y.data))


def log(x):
    return Tensor(_tinydl.op_log(x.data))


def exp(x):
    return Tensor(_tinydl.op_exp(x.data))


def relu(x):
    return Tensor(_tinydl.op_relu(x.data))

