from .. import Tensor
from collections import OrderedDict
import numpy as np
from ..import matmul
from ..import conv2d
from ..import batchnorm
from .. import _tinydl
from . import init


class Parameter(Tensor):
    def __init__(self, *arg, require_grad=True, **kwarg):
        super().__init__(*arg, **kwarg)
        self.require_grad_(require_grad)

    def set_value(self, data):
        require_grad = self.data.has_grad()
        # print("set_value to", data)
        # from IPython import embed
        # embed()
        if isinstance(data, np.ndarray):
            assert self.shape() == data.shape, "set parameter with shape {} with data shape {}".format(self.shape(), data.shape)
            t = _tinydl.Tensor(data)
            # print("set_value", self.shape(), t.to_numpy().shape)
            self.data.set_value(t)
        else:
            assert isinstance(data, Tensor)
            # print("set_value", self.shape(), data.shape())
            self.data.set_value(data.data)
        self.data.require_grad_(require_grad)



class Module:
    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
        self._is_train = True

    def train(self):
        self._is_train = True
        for m in self.children():
            m.train()

    def eval(self):
        self._is_train = False
        for m in self.children():
            m.eval()

    def parameters(self):
        parameters = list(self._parameters.values())
        for c in self.children():
            parameters.extend(c.parameters())
        return parameters

    def named_parameters(self):
        parameters =  OrderedDict(self._parameters.items())
        for name, mod in self.named_children():
            for pname, p in mod.named_parameters():
                parameters["{}.{}".format(name, pname)] = p
        return parameters.items()

    def children(self):
        children = list(self._modules.values())
        return children

    def named_children(self):
        children =  OrderedDict(self._modules.items())
        return children.items()

    def __setattr__(self, key, value):
        if isinstance(value, Parameter):
            self.__dict__["_parameters"][key] = value
        elif isinstance(value, Module):
            self.__dict__["_modules"][key] = value
        else:
            self.__dict__[key] = value
            # super().__setattr__(key, value)

    def __getattr__(self, key):
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]
        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        return self.__dict__[key]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("")

    def __call__(self, data):
        return self.forward(data)
    
    def load_state_dict(self, state):
        for k, v in self.named_parameters():
            if k in state:
                v.set_value(state[k])
        for m_name, mod in self.named_children():
            prefix = "{}.".format(m_name)
            s = {k.replace(prefix, ""): v for k, v in state.items() if k.startswith(prefix)}
            mod.load_state_dict(s)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.weight = Parameter(np.random.randn(in_features, out_features).astype(np.float32))
        if self.has_bias:
            self.bias = Parameter(np.zeros((1, out_features), dtype=np.float32))
        self.init()

    def forward(self, data):
        data = matmul(data, self.weight)
        if self.has_bias:
            data = data + self.bias
        return data

    def init(self):
        init.kaiming_uniform_(self.weight, self.bias)


class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.has_bias = bias
        self.padding = padding
        self.stride = stride
        self.weight = Parameter(
                np.empty((out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
                )
        if self.has_bias:
            self.bias = Parameter(np.zeros((1, out_channels, 1, 1), dtype=np.float32))
        self.init()

    def forward(self, data):
        data = conv2d(data, self.weight, self.padding, self.stride)
        if self.has_bias:
            data = data + self.bias
        return data

    def init(self):
        init.kaiming_uniform_(self.weight, self.bias)


class BatchNorm(Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.weight = Parameter(
                np.ones((1, channels, 1, 1)).astype(np.float32)
                )
        self.bias = Parameter(np.zeros((1, channels, 1, 1), dtype=np.float32))
        self.running_mean = Parameter(
                np.zeros((1, channels, 1, 1), dtype=np.float32),
                require_grad=False)
        self.running_var = Parameter(
                np.zeros((1, channels, 1, 1), dtype=np.float32),
                require_grad=False)

    def forward(self, data):
        return batchnorm(data, self.weight, self.bias, self.running_mean, self.running_var, self._is_train)
