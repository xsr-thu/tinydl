from .. import Tensor
from collections import OrderedDict
import numpy as np
from ..import matmul
from .. import _tinydl


class Parameter(Tensor):
    def set_value(self, data):
        assert isinstance(data, np.ndarray)
        assert self.shape() == data.shape, "set parameter with shape {} with data shape {}".format(self.shape(), data.shape)
        self.data = _tinydl.Tensor(data)



class Module:
    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    def parameters(self):
        return list(self._parameters.value())

    def named_parameters(self):
        return list(self._parameters.items())

    def named_children(self):
        return list(self._modules.items())

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

    def forward(self, data):
        data = matmul(data, self.weight)
        if self.has_bias:
            data = data + self.bias
        return data
