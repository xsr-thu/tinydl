#ifndef __CONV_H_
#define __CONV_H_

#include "tensor.h"

namespace opr {

Tensor conv2d(Tensor& data, Tensor& weight);

}

#endif
