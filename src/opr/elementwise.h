#ifndef __ELEMENTWISE_H_
#define __ELEMENTWISE_H_

#include <cuda_runtime.h>
#include <memory>
#include "tensor.h"


// TODO: fix
std::shared_ptr<TensorStorage> copy(std::shared_ptr<TensorStorage> x);
shared_ptr<TensorStorage> add(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y);
shared_ptr<TensorStorage> mul(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y);

namespace opr {
    Tensor add(Tensor& x, Tensor& y);
    Tensor sub(Tensor& x, Tensor& y);
    Tensor mul(Tensor& x, Tensor& y);
    Tensor div(Tensor& x, Tensor& y);

    Tensor equal(Tensor& x, Tensor& y);
    Tensor less_then(Tensor& x, Tensor& y);
    Tensor less_equal(Tensor& x, Tensor& y);
    Tensor greater_then(Tensor& x, Tensor& y);
    Tensor greater_equal(Tensor& x, Tensor& y);

    Tensor relu(Tensor& x);
    Tensor log(Tensor& x);
    Tensor exp(Tensor& x);
    Tensor sigmoid(Tensor& x);
    Tensor copy(Tensor& x);

    Tensor as_float32(Tensor& x);
    Tensor as_bool(Tensor& x);

namespace intl {    
    using ::copy;
    using ::add;
    using ::mul;
}

}
#endif
