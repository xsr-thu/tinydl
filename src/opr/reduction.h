#ifndef __REDUCTION_H_
#define __REDUCTION_H_

#include <cuda_runtime.h>
#include "tensor.h"

namespace opr {

Tensor reduce_sum(Tensor &input, const vector<size_t> &axis, const bool keep_dim=false);
Tensor reduce_mean(Tensor &input, const vector<size_t> &axis, const bool keep_dim=false);

namespace intl {

shared_ptr<TensorStorage> reduce_sum(shared_ptr<TensorStorage> input, const vector<size_t> &axis, const bool keep_dim=false);
shared_ptr<TensorStorage> reduce_mean(shared_ptr<TensorStorage> input, const vector<size_t> &axis, const bool keep_dim=false);

}
}

#endif
