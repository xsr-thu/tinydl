#ifndef __BATCH_NORM_H_
#define __BATCH_NORM_H_

#include "tensor.h"

namespace opr {

Tensor batchnorm(Tensor& data, Tensor &scale, Tensor &bias, Tensor &running_mean, Tensor &running_var, bool is_train);

}

#endif
