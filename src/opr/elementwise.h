#ifndef __ELEMENTWISE_H_
#define __ELEMENTWISE_H_

#include <cuda_runtime.h>
#include <memory>
#include "tensor.h"

std::shared_ptr<TensorStorage> copy(std::shared_ptr<TensorStorage> x);
shared_ptr<TensorStorage> add(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y);

namespace opr {
    Tensor add(Tensor& x, Tensor& y);
    Tensor sub(Tensor& x, Tensor& y);
    Tensor mul(Tensor& x, Tensor& y);
    Tensor div(Tensor& x, Tensor& y);

    Tensor relu(Tensor& x);
    Tensor log(Tensor& x);
    Tensor exp(Tensor& x);
    Tensor sigmoid(Tensor& x);
    Tensor copy(Tensor& x);

namespace intl {    
    using ::copy;
    using ::add;
}

}
#endif
