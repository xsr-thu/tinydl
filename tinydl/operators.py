from . import _tinydl


def _as_tensor(x):
    from .tensor import Tensor
    return Tensor(x)


def matmul(x, y):
    return _as_tensor(_tinydl.op_matmul(x.data, y.data))


def log(x):
    return _as_tensor(_tinydl.op_log(x.data))


def exp(x):
    return _as_tensor(_tinydl.op_exp(x.data))


def relu(x):
    return _as_tensor(_tinydl.op_relu(x.data))


def neg(x):
    return _as_tensor(_tinydl.op_neg(x.data))


def as_float32(x):
    return _as_tensor(_tinydl.op_as_float32(x.data))


def as_bool(x):
    return _as_tensor(_tinydl.op_as_bool(x.data))


def add(x, y):
    return _as_tensor(_tinydl.op_add(x.data, y.data))


def sub(x, y):
    return _as_tensor(_tinydl.op_sub(x.data, y.data))


def mul(x, y):
    return _as_tensor(_tinydl.op_mul(x.data, y.data))


def div(x, y):
    return _as_tensor(_tinydl.op_div(x.data, y.data))


def equal(x, y):
    return _as_tensor(_tinydl.op_equal(x.data, y.data))


def less_then(x, y):
    return _as_tensor(_tinydl.op_less_then(x.data, y.data))


def less_equal(x, y):
    return _as_tensor(_tinydl.op_less_equal(x.data, y.data))


def greater_then(x, y):
    return _as_tensor(_tinydl.op_greater_then(x.data, y.data))


def greater_equal(x, y):
    return _as_tensor(_tinydl.op_greater_equal(x.data, y.data))


def sum(x, axis, keep_dim=False):
    if isinstance(axis, int):
        axis = [axis]
    return _as_tensor(_tinydl.op_reduce_sum(x.data, axis, keep_dim))


def mean(x, axis, keep_dim=False):
    if isinstance(axis, int):
        axis = [axis]
    return _as_tensor(_tinydl.op_reduce_mean(x.data, axis, keep_dim))


def max(x, axis, keep_dim=False):
    if isinstance(axis, int):
        axis = [axis]
    return _as_tensor(_tinydl.op_reduce_max(x.data, axis, keep_dim))


def min(x, axis, keep_dim=False):
    if isinstance(axis, int):
        axis = [axis]
    return _as_tensor(_tinydl.op_reduce_min(x.data, axis, keep_dim))


def view(x, shape):
    return _as_tensor(_tinydl.op_view(x.data, shape))


def conv2d(x, w, padding, stride):
    return _as_tensor(_tinydl.op_conv2d(x.data, w.data, padding, stride))


def batchnorm(x, weight, bias, running_mean, running_var, is_train):
    return _as_tensor(_tinydl.op_batchnorm(
        x.data, weight.data, bias.data,
        running_mean.data, running_var.data,
        is_train))
