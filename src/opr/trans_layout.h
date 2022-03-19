#ifndef __TRANS_LAYOUT_H_
#define __TRANS_LAYOUT_H_

#include "tensor.h"

namespace opr {
    Tensor view(Tensor& x, std::vector<size_t> &new_shape);
}


#endif
