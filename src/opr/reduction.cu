#include <cassert>
#include "reduction.h"
#include "opr_utils.h"
#include "elementwise.h"

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


struct ReductionBackwardFunc: BackwardFunc {
    vector<size_t> m_old_shape;
    bool m_keep_dim;
    ReductionMode m_mode;
    size_t m_reduction_size;

    static std::shared_ptr<BackwardFunc> make(const ReductionMode mode, size_t reduction_size, shared_ptr<GraphNode> x, std::vector<size_t> &shape, bool keep_dim){
        shared_ptr<ReductionBackwardFunc> func = make_shared<ReductionBackwardFunc>();
        func->m_input_nodes.push_back(x);
        func->m_old_shape = shape;
        func->m_keep_dim = keep_dim;
        func->m_mode = mode;
        func->m_reduction_size = reduction_size;
        return func;
    }

    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();

        if (m_mode == ReductionMode::MEAN) {
            out_grad = opr::intl::mul(out_grad, Tensor::get_const(1.0/(m_reduction_size)).storage());
        }

        if(!m_keep_dim) {
            vector<size_t> old_strides(m_old_shape.size());
            size_t s = sizeof(float);
            for(int i=m_old_shape.size() - 1; i>=0; --i) {
                old_strides[i] = s;
                s *= m_old_shape[i];
            }

            shared_ptr<TensorStorage> new_storage = make_shared<TensorStorage>(
                    out_grad, m_old_shape, old_strides);
            m_input_nodes[0]->acc_grad(new_storage);
        } else {
            m_input_nodes[0]->acc_grad(out_grad);
        }
    }
};

shared_ptr<TensorStorage> reduction(const ReductionMode mode, shared_ptr<TensorStorage> input, const vector<size_t> &axis, const bool keep_dim) {
    vector<size_t> output_shape = input->shape();
    vector<size_t> output_strides(output_shape.size());
    size_t output_size = 1;
    bool keep[MAX_DIM]={};
    memset(keep, true, MAX_DIM);
    for(size_t i =0;i<axis.size(); i++) {
        output_shape[axis[i]] = 1;
        keep[axis[i]] = false;
    }
    for(int i=output_shape.size() - 1; i>= 0; --i) {
        output_strides[i] = output_size * sizeof(float);
        output_size *= output_shape[i];
    }
    float *res;
    cudaMalloc(&res, sizeof(float) * output_size);

    TensorFormat *in_format = TensorFormat::make_cuda_tensor_format(input);
    TensorFormat *out_format = TensorFormat::make_cuda_tensor_format(output_shape, output_strides);

    int block_size = 128;
    int n_block = (output_size + block_size - 1) / block_size;
    kernel_reduction_op<<<n_block, block_size>>>(res, out_format, input->data(), in_format, mode);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error\n");
    }
    out_format->release();
    in_format->release();

    vector<size_t> out_shape;
    vector<size_t> out_strides;
    for(size_t i=0; i<output_shape.size(); i++) {
        if(keep[i]) {
            out_shape.push_back(output_shape[i]);
            out_strides.push_back(output_strides[i]);
        }
    }

    // TODO: check 0 dim tensor
    Tensor res_tensor;

    if(keep_dim) {
        // fprintf(stderr, "reduction --> shape %s stride %s\n", to_string(output_shape).c_str(), to_string(output_strides).c_str());
        return make_shared<TensorStorage>(res, output_size, output_shape, output_strides);
    } else {
        // fprintf(stderr, "reduction --> shape %s stride %s\n", to_string(out_shape).c_str(), to_string(output_strides).c_str());
        return make_shared<TensorStorage>(res, output_size, out_shape, out_strides);
    }
}


Tensor reduction(const ReductionMode mode, Tensor &input, const vector<size_t> &axis, const bool keep_dim) {
    Tensor res_tensor(reduction(mode, input.storage(), axis, keep_dim));
    // fprintf(stderr, "setting tensor %zu\n", res_tensor.m_id);
    if(input.need_grad()) {
        shared_ptr<GraphNode> out_node = res_tensor.graph_node();
        shared_ptr<GraphNode> input_node = input.graph_node();

        vector<size_t> output_shape = input.storage()->shape();
        size_t reduction_size = 1;
        for(size_t i =0;i<axis.size(); i++) {
            reduction_size *= output_shape[axis[i]];
            output_shape[axis[i]] = 1;
        }

        shared_ptr<BackwardFunc> func = ReductionBackwardFunc::make(mode, reduction_size, input_node, output_shape, keep_dim);

        out_node->set_backward_func(func);
        // fprintf(stderr, "setting tensor %zu -- add backward\n", res_tensor.m_id);
    }
    return res_tensor;
}

namespace opr{

Tensor reduce_sum(Tensor &input, const vector<size_t> &axis, const bool keep_dim) {
    return reduction(ReductionMode::SUM, input, axis, keep_dim);
}


Tensor reduce_mean(Tensor &input, const vector<size_t> &axis, const bool keep_dim) {
    return reduction(ReductionMode::MEAN, input, axis, keep_dim);
}

namespace intl {


shared_ptr<TensorStorage> reduce_sum(shared_ptr<TensorStorage> input, const vector<size_t> &axis, const bool keep_dim) {
    return reduction(ReductionMode::SUM, input, axis, keep_dim);
}

shared_ptr<TensorStorage> reduce_mean(shared_ptr<TensorStorage> input, const vector<size_t> &axis, const bool keep_dim) {
    return reduction(ReductionMode::MEAN, input, axis, keep_dim);
}

}
}
