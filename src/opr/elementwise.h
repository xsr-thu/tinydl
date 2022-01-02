#ifndef __ELEMENTWISE_H_
#define __ELEMENTWISE_H_

#include <cuda_runtime.h>
#include "tensor.h"
namespace opr {

    Tensor add(Tensor x, Tensor y);
    Tensor sub(Tensor x, Tensor y);
    Tensor mul(Tensor x, Tensor y);
    Tensor div(Tensor x, Tensor y);

    Tensor relu(Tensor x);
    Tensor log(Tensor x);
    Tensor exp(Tensor x);
    Tensor sigmoid(Tensor x);

}
#endif
