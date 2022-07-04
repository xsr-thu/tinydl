#include <stdio.h>
#include <cassert>
#include "common.h"

void cudnn_check(cudnnStatus_t s, std::string file, size_t line) {
    if(s != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "CUDNN: %s (%s:%lu)\n", cudnnGetErrorString(s), file.c_str(), line);
        assert(0);
    }
}
