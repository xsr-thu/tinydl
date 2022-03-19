#ifndef __REDUCTION_H_
#define __REDUCTION_H_

#include <cuda_runtime.h>
#include "tensor.h"

namespace opr {

Tensor reduce_sum(Tensor &input, const vector<size_t> &axis, const bool keep_dim=false);
Tensor reduce_mean(Tensor &input, const vector<size_t> &axis, const bool keep_dim=false);

}

#endif
