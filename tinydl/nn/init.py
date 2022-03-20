import numpy as np
import math


def kaiming_uniform_(weight, bias):
    fan_in = weight.shape()[0]
    bound = math.sqrt(2. * 3. / fan_in)
    w = np.random.uniform(-bound, bound, weight.shape())
    weight.set_value(w)
    if bias is not None:
        b = 1. / math.sqrt(fan_in)
        b = np.random.uniform(-b, b, bias.shape())
        bias.set_value(b)
