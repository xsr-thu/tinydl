#include <stdio.h>
#include <cassert>
#include "common.h"

void cudnn_check(cudnnStatus_t s) {
    if(s != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "CUDNN: %s\n", cudnnGetErrorString(s));
        assert(0);
    }
}
