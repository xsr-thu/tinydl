#include <cassert>
#include "reduction.h"
#include "opr_utils.h"

enum class ReductionMode {
    UNDEFINED,
    SUM,
    MEAN,
};

#define MAX_DIM 7

__global__ void kernel_reduction_op(float *out, TensorFormat *out_format, float* in, TensorFormat *in_format, ReductionMode mode) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= out_format->size)
        return;

    // ordinal to index
    size_t out_idx[MAX_DIM];
    size_t ordinal = idx;
    for(int i=out_format->dim-1;i>=0;i--) {
        out_idx[i] = ordinal % out_format->shape[i];
        // printf(">idx=%lu, out_idx[%d]=%lu ordinal=%lu, \n", idx, i, out_idx[i], ordinal);
        ordinal = ordinal / out_format->shape[i];
    }
    float ans = 0.f;
    
    size_t reduction_size = 1;
    size_t reduction_dims[MAX_DIM];
    size_t reduction_dim = 0;
    for(int i=0;i<out_format->dim; i++) {
        if(out_format->shape[i] != in_format->shape[i]) {
            reduction_size *= in_format->shape[i];
            reduction_dims[reduction_dim] = i;
            reduction_dim ++;
        }
    }
    for(size_t i=0; i<reduction_size; i++) {
        size_t tmp_idx = i;
        for(int j=reduction_dim - 1; j >=0; j--) {
            out_idx[reduction_dims[j]] = tmp_idx % in_format->shape[reduction_dims[j]];
            tmp_idx = tmp_idx / in_format->shape[reduction_dims[j]];
        }
        // to index
        size_t in_ordinal = 0;
        for(size_t j=0; j<in_format->dim; j++) {
            in_ordinal += out_idx[j] * in_format->strides[j] / sizeof(float);
            // printf("  idx=%d out_idx[%lu]=%lu, strides[%lu]=%lu --> %lu\n", idx, j, out_idx[j], j, in_format->strides[j], in_ordinal);
        }
        ans = ans + in[in_ordinal];
    }
    if(mode == ReductionMode::MEAN) {
        ans = ans / reduction_size;
    }
    out[idx] = ans;
}

Tensor reduction(const ReductionMode mode, const Tensor &input, const vector<size_t> &axis, const bool keep_dim) {
    vector<size_t> output_shape = input.m_storage->m_shape;
    vector<size_t> output_strides;
    size_t output_size = sizeof(float);
    bool keep[MAX_DIM]={};
    memset(keep, true, MAX_DIM);
    for(size_t i =0;i<axis.size(); i++) {
        output_shape[axis[i]] = 1;
        keep[axis[i]] = false;
    }
    for(size_t i=0;i<output_shape.size();i++) {
        output_strides.push_back(output_size);
        output_size *= output_shape[i];
    }
    float *res;
    cudaMalloc(&res, sizeof(float) * output_size);

    TensorFormat *in_format = TensorFormat::make_cuda_tensor_format(input);
    TensorFormat *out_format = TensorFormat::make_cuda_tensor_format(output_shape, output_strides);

    int block_size = 128;
    int n_block = (output_size + block_size - 1) / block_size;
    kernel_reduction_op<<<n_block, block_size>>>(res, out_format, input.m_storage->data(), in_format, mode);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error\n");
    }
    out_format->release();
    in_format->release();
    if(keep_dim)
        return Tensor(res, output_shape, output_strides);
    vector<size_t> out_shape;
    vector<size_t> out_strides;
    for(size_t i=0; i<output_shape.size(); i++) {
        if(keep[i]) {
            out_shape.push_back(output_shape[i]);
            out_strides.push_back(output_strides[i]);
        }
    }
    return Tensor(res, out_shape, out_strides);
}

namespace opr{

Tensor reduce_sum(const Tensor &input, const vector<size_t> &axis, const bool keep_dim) {
    return reduction(ReductionMode::SUM, input, axis, keep_dim);
}


Tensor reduce_mean(const Tensor &input, const vector<size_t> &axis, const bool keep_dim) {
    return reduction(ReductionMode::MEAN, input, axis, keep_dim);
}

}
