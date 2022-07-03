#include <cassert>
#include "tensor.h"
#include "elementwise.h"
#include "reduction.h"
#include "opr_utils.h"
#include "dtype.h"

using namespace std;

struct BinaryOpBackwarFuncBase;

// *****************************************************************************
template<typename DT>
struct AsFloat32Op {
    using DType = DT;
    using T = typename DT::T;
    using RT = float;
    static __device__ __forceinline__ RT apply(T x) {
        return static_cast<RT>(x);
    }
};

template<typename DT>
struct AsBoolOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = bool;
    static __device__ __forceinline__ RT apply(T x) {
        return static_cast<RT>(x);
    }
};

// *****************************************************************************
template<typename DT>
struct ReluOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = typename DT::T;
    static __device__ __forceinline__ RT apply(T x) {
        return x > DT::one() ? x: DT::zero();
    }
};

template<typename DT>
struct ExpOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = typename DT::T;
    static __device__ __forceinline__ RT apply(T x) {
        return DT::exp(x);
    }
};

template<typename DT>
struct LogOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = typename DT::T;
    static __device__ __forceinline__ RT apply(T x) {
        return DT::log(x);
    }
};

template<typename DT>
struct SigmoidOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = typename DT::T;
    static __device__ __forceinline__ RT apply(T x) {
        return DT::one() / (DT::one() + DT::exp(x));
    }
};

template<typename DT>
struct NegOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = typename DT::T;
    static __device__ __forceinline__ RT apply(T x) {
        return -x;
    }
};

template<typename DT>
struct CopyOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = typename DT::T;
    static __device__ __forceinline__ RT apply(T x) {
        return x;
    }
};

template<typename DT>
struct ReciprocalOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = typename DT::T;
    static __device__ __forceinline__ RT apply(T x) {
        return DT::one() / x;
    }
};


template<typename DT>
struct AddOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = typename DT::T;
    static __device__ __forceinline__ RT apply(T lhs, T rhs) {
        return lhs + rhs;
    }
};

template<typename DT>
struct SubOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = typename DT::T;
    static __device__ __forceinline__ RT apply(T lhs, T rhs) {
        return lhs - rhs;
    }
};

template<typename DT>
struct MulOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = typename DT::T;
    static __device__ __forceinline__ RT apply(T lhs, T rhs) {
        return lhs * rhs;
    }
};

template<typename DT>
struct DivOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = typename DT::T;
    static __device__ __forceinline__ RT apply(T lhs, T rhs) {
        return lhs / rhs;
    }
};

// *****************************************************************************
template<typename DT>
struct BooleanEqualOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = bool;
    static __device__ __forceinline__ RT apply(T lhs, T rhs) {
        return lhs == rhs;
    }
};

template<typename DT>
struct BooleanLessThenOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = bool;
    static __device__ __forceinline__ RT apply(T lhs, T rhs) {
        return lhs < rhs;
    }
};

template<typename DT>
struct BooleanLessEqualOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = bool;
    static __device__ __forceinline__ RT apply(T lhs, T rhs) {
        return lhs <= rhs;
    }
};

template<typename DT>
struct BooleanGreaterThenOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = bool;
    static __device__ __forceinline__ RT apply(T lhs, T rhs) {
        return lhs > rhs;
    }
};

template<typename DT>
struct BooleanGreaterEqualOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = bool;
    static __device__ __forceinline__ RT apply(T lhs, T rhs) {
        return lhs >= rhs;
    }
};
// *****************************************************************************
template<typename DT>
struct EqualOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = T;
    static __device__ __forceinline__ RT apply(T lhs, T rhs) {
        return lhs == rhs;
    }
};

template<typename DT>
struct LessThenOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = T;
    static __device__ __forceinline__ RT apply(T lhs, T rhs) {
        return lhs < rhs;
    }
};

template<typename DT>
struct LessEqualOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = T;
    static __device__ __forceinline__ RT apply(T lhs, T rhs) {
        return lhs <= rhs;
    }
};

template<typename DT>
struct GreaterThenOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = T;
    static __device__ __forceinline__ RT apply(T lhs, T rhs) {
        return lhs > rhs;
    }
};

template<typename DT>
struct GreaterEqualOp {
    using DType = DT;
    using T = typename DT::T;
    using RT = T;
    static __device__ __forceinline__ RT apply(T lhs, T rhs) {
        return lhs >= rhs;
    }
};


// *****************************************************************************
template<typename Op>
__global__ void kernel_binary_op(
        typename Op::RT *out,
        typename Op::T *a,
        typename Op::T *b, size_t n) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n) {
        out[idx] = Op::apply(a[idx], b[idx]);
    }
}


template<typename Op>
__global__ void kernel_binary_op(
        typename Op::RT *out, TensorFormat *out_format,
        typename Op::T *a, TensorFormat* a_format,
        typename Op::T *b, TensorFormat* b_format,
        size_t n) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    int indices[TensorFormat::MAX_DIM];

    for(int idx0=idx, i=out_format->dim-1; i>=0; i--) {
        indices[i] = idx0 % out_format->shape[i];
        idx0 = idx0 / out_format->shape[i];
    }
    
    size_t idx_a = 0, idx_b = 0;
    for(int i=0; i<out_format->dim; i++) {
        size_t sa = a_format->shape[i] == 1 ? 0: (a_format->strides[i]);
        size_t sb = b_format->shape[i] == 1 ? 0: (b_format->strides[i]);
        idx_a  += sa * indices[i];
        idx_b  += sb * indices[i];
    }

    if(idx < n) {
        out[idx] = Op::apply(a[idx_a], b[idx_b]);
    }
}


template<typename Op>
__global__ void kernel_unary_op(typename Op::RT *out, typename Op::T *in, size_t n) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n) {
        out[idx] = Op::apply(in[idx]);
    }
}

// *****************************************************************************
template<typename Op>
shared_ptr<TensorStorage> binary_op(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    bool same_layout = true;

    same_layout = same_layout && (x->size() == y->size());
    same_layout = same_layout && (x->dim() == y->dim());
    for(size_t i=0; i<x->dim(); i++)
        same_layout = same_layout && (x->shape()[i] == y->shape()[i]) && (x->strides()[i] == y->strides()[i]);

    typename Op::RT *res;
    if(same_layout) {
        cudaMalloc(&res, sizeof(typename Op::RT) * x->size());

        int block_size = 128;
        int n_block = (x->size() + block_size - 1) / block_size;
        using DType = typename Op::T;
        kernel_binary_op<Op><<<n_block, block_size>>>(res, x->data<DType>(), y->data<DType>(), x->size());
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("cuda error %s\n", cudaGetErrorString(err));
        }
        return make_shared<TensorStorage>(res, x->size(), x->shape(), x->strides(),
                typeclass_to_enum<typename Op::RT>());
    } else {
        vector<size_t> out_shape;
        vector<size_t> out_strides;
        std::shared_ptr<TensorFormat> x_format, y_format, out_format;
        size_t res_size;

        if(x->dim() == y->dim()) {
            out_shape.resize(x->dim());
            out_strides.resize(x->dim());
            size_t s = 1;
            for(int i=x->dim() - 1;i >= 0; i--) {
                out_shape[i] = max(x->shape()[i], y->shape()[i]);
                out_strides[i] = s;
                s *= out_shape[i];
            }
            res_size = s;
            x_format = TensorFormat::make_cuda_tensor_format(x->shape(), x->strides());
            y_format = TensorFormat::make_cuda_tensor_format(y->shape(), y->strides());
            out_format = TensorFormat::make_cuda_tensor_format(out_shape, out_strides);
        } else if(x->dim() == 1 && x->size() ==1) {
            vector<size_t> x_shape;
            vector<size_t> x_strides;
            x_shape.resize(y->dim());
            x_strides.resize(y->dim());
            out_shape.resize(y->dim());
            out_strides.resize(y->dim());
            size_t s = 1;
            for(int i=y->dim() - 1;i >= 0; i--) {
                out_shape[i] = y->shape()[i];
                out_strides[i] = s;
                s *= out_shape[i];
                x_shape[i] = 1;
                x_strides[i] = 1;
            }
            res_size = s;
            x_format = TensorFormat::make_cuda_tensor_format(x_shape, x_strides);
            y_format = TensorFormat::make_cuda_tensor_format(y->shape(), y->strides());
            out_format = TensorFormat::make_cuda_tensor_format(out_shape, out_strides);
        } else if(y->dim() == 1 && y->size() == 1) {
            vector<size_t> y_shape;
            vector<size_t> y_strides;
            y_shape.resize(x->dim());
            y_strides.resize(x->dim());
            out_shape.resize(x->dim());
            out_strides.resize(x->dim());
            size_t s = 1;
            for(int i=x->dim() - 1;i >= 0; i--) {
                out_shape[i] = x->shape()[i];
                out_strides[i] = s;
                s *= out_shape[i];
                y_shape[i] = 1;
                y_strides[i] = 1;
            }
            res_size = s;
            x_format = TensorFormat::make_cuda_tensor_format(x->shape(), x->strides());
            y_format = TensorFormat::make_cuda_tensor_format(y_shape, y_strides);
            out_format = TensorFormat::make_cuda_tensor_format(out_shape, out_strides);
        } else {
            fprintf(stderr, "binary op error\n");
            throw "Error";
        }
        
        cudaMalloc(&res, sizeof(typename Op::RT) * res_size);

        int block_size = 128;
        int n_block = (res_size + block_size - 1) / block_size;

        using DType = typename Op::T;
        kernel_binary_op<Op><<<n_block, block_size>>>(res, out_format.get(),
                x->data<DType>(), x_format.get(),
                y->data<DType>(), y_format.get(),
                res_size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("cuda error %s\n", cudaGetErrorString(err));
        }
        return make_shared<TensorStorage>(res, res_size, out_shape, out_strides,
                typeclass_to_enum<typename Op::RT>());
    }
}


template<template<typename> typename Op>
shared_ptr<TensorStorage> dispatch_binary_op(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    // TODO: type check
    switch(x->dtype()) {
        case DataType::Float32:
            return binary_op<Op<Float32>>(x, y);
        case DataType::UInt64:
            return binary_op<Op<UInt64>>(x, y);
        case DataType::Bool:
            return binary_op<Op<Bool>>(x, y);
        default:
            throw "Excepton";
    }
}



template<typename Op>
shared_ptr<TensorStorage> unary_op(shared_ptr<TensorStorage> x) {
    typename Op::RT *res;
    cudaMalloc(&res, sizeof(typename Op::RT) * x->size());

    int block_size = 128;
    int n_block = (x->size() + block_size - 1) / block_size;
    using DType = typename Op::T;
    kernel_unary_op<Op><<<n_block, block_size>>>(res, x->data<DType>(), x->size());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cuda error %s\n", cudaGetErrorString(err));
    }
    return make_shared<TensorStorage>(res, x->size(), x->shape(), x->strides(),
            typeclass_to_enum<typename Op::RT>());
}


template<template<typename> typename Op>
shared_ptr<TensorStorage> dispatch_unary_op(shared_ptr<TensorStorage> x) {
    switch(x->dtype()) {
        case DataType::Float32:
            return unary_op<Op<Float32>>(x);
        case DataType::UInt64:
            return unary_op<Op<UInt64>>(x);
        case DataType::Bool:
            return unary_op<Op<Bool>>(x);
        default:
            throw "Excepton";
    }
}


template<typename T>
shared_ptr<TensorStorage> copy_op(shared_ptr<TensorStorage> x) {
    T *res;
    // FIXME: multiply sizeof(float)??
    cudaMalloc(&res, sizeof(T) * x->size());

    cudaMemcpy(res, x->data(), sizeof(T) * x->size(), cudaMemcpyDeviceToDevice);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cuda error %s\n", cudaGetErrorString(err));
    }
    return make_shared<TensorStorage>(res, x->size(), x->shape(), x->strides(),
            typeclass_to_enum<T>());
}


// *****************************************************************************

shared_ptr<TensorStorage> add(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return dispatch_binary_op<AddOp>(x, y);
}

shared_ptr<TensorStorage> sub(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return dispatch_binary_op<SubOp>(x, y);
}

shared_ptr<TensorStorage> mul(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return dispatch_binary_op<MulOp>(x, y);
}

shared_ptr<TensorStorage> div(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return dispatch_binary_op<DivOp>(x, y);
}

shared_ptr<TensorStorage> equal(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return dispatch_binary_op<EqualOp>(x, y);
}

shared_ptr<TensorStorage> less_then(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return dispatch_binary_op<LessThenOp>(x, y);
}

shared_ptr<TensorStorage> less_equal(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return dispatch_binary_op<LessEqualOp>(x, y);
}

shared_ptr<TensorStorage> greater_then(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return dispatch_binary_op<GreaterThenOp>(x, y);
}

shared_ptr<TensorStorage> greater_equal(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return dispatch_binary_op<GreaterEqualOp>(x, y);
}

shared_ptr<TensorStorage> relu(shared_ptr<TensorStorage> x) {
    return dispatch_unary_op<ReluOp>(x);
}

shared_ptr<TensorStorage> exp(shared_ptr<TensorStorage> x) {
    // FIXME:
    return unary_op<ExpOp<Float32>>(x);
}

shared_ptr<TensorStorage> log(shared_ptr<TensorStorage> x) {
    // FIXME:
    return unary_op<LogOp<Float32>>(x);
}

shared_ptr<TensorStorage> sigmoid(shared_ptr<TensorStorage> x) {
    // FIXME:
    return unary_op<SigmoidOp<Float32>>(x);
}

shared_ptr<TensorStorage> neg(shared_ptr<TensorStorage> x) {
    return dispatch_unary_op<NegOp>(x);
}


shared_ptr<TensorStorage> as_float32(shared_ptr<TensorStorage> x) {
    return dispatch_unary_op<AsFloat32Op>(x);
}


shared_ptr<TensorStorage> as_bool(shared_ptr<TensorStorage> x) {
    return dispatch_unary_op<AsBoolOp>(x);
}


shared_ptr<TensorStorage> copy(shared_ptr<TensorStorage> x) {
    switch(x->dtype()) {
        case DataType::Float32:
            return copy_op<typename Float32::T>(x);
        case DataType::UInt64:
            return copy_op<typename UInt64::T>(x);
        case DataType::Bool:
            return copy_op<typename Bool::T>(x);
        default:
            return nullptr;
    }
}

shared_ptr<TensorStorage> reciprocal(shared_ptr<TensorStorage> x) {
    return unary_op<ReciprocalOp<Float32>>(x);
}

// *****************************************************************************
struct BinaryOpBackwarFuncBase: BackwardFunc {
    vector<size_t> m_x_shape;
    vector<size_t> m_y_shape;
    template<typename Op>
    static std::shared_ptr<BackwardFunc> make(
            shared_ptr<GraphNode> x, const vector<size_t> &x_shape,
            shared_ptr<GraphNode> y, const vector<size_t> &y_shape);

    BinaryOpBackwarFuncBase(const vector<size_t> &x_shape, const vector<size_t> &y_shape)
        : m_x_shape(x_shape), m_y_shape(y_shape) {}
};

template<typename DT, typename Op>
struct BinaryOpBackwarFunc: BinaryOpBackwarFuncBase {
    BinaryOpBackwarFunc(const vector<size_t> &x_shape, const vector<size_t> &y_shape)
        :BinaryOpBackwarFuncBase(x_shape, y_shape) {};

    void backward_func(shared_ptr<GraphNode> out_node) override {
        // FIXME: not implemented for GT GE etc.
    }
};


template<typename DT>
struct BinaryOpBackwarFunc<DT, AddOp<DT>>: BinaryOpBackwarFuncBase {
    BinaryOpBackwarFunc(const vector<size_t> &x_shape, const vector<size_t> &y_shape)
        :BinaryOpBackwarFuncBase(x_shape, y_shape) {};

    void backward_func(shared_ptr<GraphNode> out_node) override {
        // TODO: check shape
        vector<size_t> x_reduce_dims;
        vector<size_t> y_reduce_dims;
        assert(out_node->dim() == m_x_shape.size());
        assert(out_node->dim() == m_y_shape.size());

        for(int i=0; i<out_node->dim(); i++) {
            if(out_node->shape()[i] != m_x_shape[i]) {
                assert(m_x_shape[i] == 1);
                x_reduce_dims.push_back(i);
            }
            if(out_node->shape()[i] != m_y_shape[i]) {
                assert(m_y_shape[i] == 1);
                y_reduce_dims.push_back(i);
            }
        }
        // FIXME: if inputs do not need grad, does not backprop

        if(x_reduce_dims.size()) {
            m_input_nodes[0]->acc_grad(
                    opr::intl::reduce_sum(out_node->grad_storage(), x_reduce_dims, true));
        } else {
            m_input_nodes[0]->acc_grad(out_node->grad_storage());
        }

        if(y_reduce_dims.size()) {
            m_input_nodes[1]->acc_grad(
                    opr::intl::reduce_sum(out_node->grad_storage(), y_reduce_dims, true));
        } else {
            m_input_nodes[1]->acc_grad(out_node->grad_storage());
        }
    }
};


template<typename DT>
struct BinaryOpBackwarFunc<DT, SubOp<DT>>: BinaryOpBackwarFuncBase {
    BinaryOpBackwarFunc(const vector<size_t> &x_shape, const vector<size_t> &y_shape)
        :BinaryOpBackwarFuncBase(x_shape, y_shape) {};

    void backward_func(shared_ptr<GraphNode> out_node) override {
        // TODO: check shape
        vector<size_t> x_reduce_dims;
        vector<size_t> y_reduce_dims;
        assert(out_node->dim() == m_x_shape.size());
        assert(out_node->dim() == m_y_shape.size());

        for(int i=0; i<out_node->dim(); i++) {
            if(out_node->shape()[i] != m_x_shape[i]) {
                assert(m_x_shape[i] == 1);
                x_reduce_dims.push_back(i);
            }
            if(out_node->shape()[i] != m_y_shape[i]) {
                assert(m_y_shape[i] == 1);
                y_reduce_dims.push_back(i);
            }
        }
        // FIXME: if inputs do not need grad, does not backprop
        if(x_reduce_dims.size()) {
            m_input_nodes[0]->acc_grad(
                    opr::intl::reduce_sum(out_node->grad_storage(), x_reduce_dims, true));
        } else {
            m_input_nodes[0]->acc_grad(out_node->grad_storage());
        }

        if(y_reduce_dims.size()) {
            m_input_nodes[1]->acc_grad(
                    opr::intl::reduce_sum(
                        unary_op<NegOp<DT>>(out_node->grad_storage()),
                        y_reduce_dims, true));
        } else {
            m_input_nodes[1]->acc_grad(
                    unary_op<NegOp<DT>>(out_node->grad_storage()));
        }

    }
};

template<typename DT>
struct BinaryOpBackwarFunc<DT, MulOp<DT>>: BinaryOpBackwarFuncBase {
    BinaryOpBackwarFunc(const vector<size_t> &x_shape, const vector<size_t> &y_shape)
        :BinaryOpBackwarFuncBase(x_shape, y_shape) {};

    void backward_func(shared_ptr<GraphNode> out_node) override {
        // TODO: check shape
        vector<size_t> x_reduce_dims;
        vector<size_t> y_reduce_dims;
        assert(out_node->dim() == m_x_shape.size());
        assert(out_node->dim() == m_y_shape.size());

        for(int i=0; i<out_node->dim(); i++) {
            if(out_node->shape()[i] != m_x_shape[i]) {
                assert(m_x_shape[i] == 1);
                x_reduce_dims.push_back(i);
            }
            if(out_node->shape()[i] != m_y_shape[i]) {
                assert(m_y_shape[i] == 1);
                y_reduce_dims.push_back(i);
            }
        }
        // FIXME: if inputs do not need grad, does not backprop
        if(x_reduce_dims.size()) {
            // fprintf(stderr, " backprop with reduction x n=%zu first=%zu\n", x_reduce_dims.size(), x_reduce_dims[0]);
            m_input_nodes[0]->acc_grad(
                    opr::intl::reduce_sum(
                        binary_op<MulOp<DT>>(out_node->grad_storage(), m_saved_tensors[1]),
                        x_reduce_dims, true));
        } else {
            m_input_nodes[0]->acc_grad(
                    binary_op<MulOp<DT>>(out_node->grad_storage(), m_saved_tensors[1]));
        }

        if(y_reduce_dims.size()) {
            // fprintf(stderr, " backprop with reduction y n=%zu first=%zu\n", y_reduce_dims.size(), y_reduce_dims[0]);
            m_input_nodes[1]->acc_grad(
                    opr::intl::reduce_sum(
                        binary_op<MulOp<DT>>(out_node->grad_storage(), m_saved_tensors[0]),
                        y_reduce_dims, true));
        } else {
            m_input_nodes[1]->acc_grad(
                    binary_op<MulOp<DT>>(out_node->grad_storage(), m_saved_tensors[0]));
        }

    }
};

template<typename DT>
struct BinaryOpBackwarFunc<DT, DivOp<DT>>: BinaryOpBackwarFuncBase {
    BinaryOpBackwarFunc(const vector<size_t> &x_shape, const vector<size_t> &y_shape)
        :BinaryOpBackwarFuncBase(x_shape, y_shape) {};

    void backward_func(shared_ptr<GraphNode> out_node) override {
        // TODO: check shape
        vector<size_t> x_reduce_dims;
        vector<size_t> y_reduce_dims;
        assert(out_node->dim() == m_x_shape.size());
        assert(out_node->dim() == m_y_shape.size());

        for(int i=0; i<out_node->dim(); i++) {
            if(out_node->shape()[i] != m_x_shape[i]) {
                assert(m_x_shape[i] == 1);
                x_reduce_dims.push_back(i);
            }
            if(out_node->shape()[i] != m_y_shape[i]) {
                assert(m_y_shape[i] == 1);
                y_reduce_dims.push_back(i);
            }
        }
        // FIXME: if inputs do not need grad, does not backprop
        if(x_reduce_dims.size()) {
            // fprintf(stderr, " backprop with reduction x n=%zu first=%zu\n", x_reduce_dims.size(), x_reduce_dims[0]);
            m_input_nodes[0]->acc_grad(
                    opr::intl::reduce_sum(
                        div(out_node->grad_storage(), m_saved_tensors[1]),
                        x_reduce_dims, true));
        } else {
            m_input_nodes[0]->acc_grad(
                    div(out_node->grad_storage(), m_saved_tensors[1]));
        }

        if(y_reduce_dims.size()) {
            // fprintf(stderr, " backprop with reduction y n=%zu first=%zu\n", y_reduce_dims.size(), y_reduce_dims[0]);
            m_input_nodes[1]->acc_grad(
                    opr::intl::reduce_sum(
                        neg(div(mul(out_node->grad_storage(), m_saved_tensors[0]),
                                mul(m_saved_tensors[1], m_saved_tensors[1]))),
                        y_reduce_dims, true));
        } else {
            m_input_nodes[1]->acc_grad(
                    neg(div(mul(out_node->grad_storage(), m_saved_tensors[0]),
                            mul(m_saved_tensors[1], m_saved_tensors[1]))));
        }


    }
};


template<typename Op>
std::shared_ptr<BackwardFunc> BinaryOpBackwarFuncBase::make(
        shared_ptr<GraphNode> x, const vector<size_t> &x_shape,
        shared_ptr<GraphNode> y, const vector<size_t> &y_shape){
    using DType = typename Op::DType;
    shared_ptr<BackwardFunc> func = make_shared<BinaryOpBackwarFunc<DType, Op>>(x_shape, y_shape);
    func->m_input_nodes.push_back(x);
    func->m_input_nodes.push_back(y);
    return func;
}


template<typename Op>
Tensor binary_op(Tensor& x, Tensor& y) {
    Tensor res = Tensor(binary_op<Op>(x.storage(), y.storage()));
    if(x.need_grad() || y.need_grad()) {
        shared_ptr<GraphNode> x_node = x.graph_node();
        shared_ptr<GraphNode> y_node = y.graph_node();
        shared_ptr<GraphNode> out_node = res.graph_node();
        shared_ptr<BackwardFunc> func = BinaryOpBackwarFuncBase::make<Op>(x_node, x.shape(), y_node, y.storage()->shape());
        using DType = typename Op::DType;
        if(std::is_same<Op, MulOp<DType>>::value || std::is_same<Op, DivOp<DType>>::value) {
            func->m_saved_tensors.push_back(x.storage());
            func->m_saved_tensors.push_back(y.storage());
        }
        out_node->set_backward_func(func);
    }
    return res;
}


template<template<typename> typename Op>
Tensor dispatch_binary_op(Tensor& x, Tensor& y) {
    // TODO: type check
    switch(x.dtype()) {
        case DataType::Float32:
            return binary_op<Op<Float32>>(x, y);
        case DataType::UInt64:
            return binary_op<Op<UInt64>>(x, y);
        case DataType::Bool:
            return binary_op<Op<Bool>>(x, y);
        default:
            throw "Excepton";
    }
}



// *****************************************************************************
struct UnaryBackwardFuncBase;
template<typename DT, typename Opr>
struct UnaryBackwardFuncImpl;



struct UnaryBackwardFuncBase: BackwardFunc {
    template<typename Op>
    static std::shared_ptr<BackwardFunc> make(shared_ptr<GraphNode> x){
        using DType = typename Op::DType;
        shared_ptr<BackwardFunc> func = make_shared<UnaryBackwardFuncImpl<DType, Op>>();
        func->m_input_nodes.push_back(x);
        return func;
    }
    void backward_func(shared_ptr<GraphNode> out_node) {
    }
};


template<typename DT, typename Opr>
struct UnaryBackwardFuncImpl: UnaryBackwardFuncBase {
};


template<typename DT>
struct UnaryBackwardFuncImpl<DT, ReluOp<DT>>: UnaryBackwardFuncBase {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        shared_ptr<TensorStorage> inp = m_saved_tensors[0];
        shared_ptr<TensorStorage> res = mul(greater_then(inp, Tensor::get_const(0.).storage()), out_grad);
        m_input_nodes[0]->acc_grad(res);
    }
};

template<typename DT>
struct UnaryBackwardFuncImpl<DT, ExpOp<DT>>: UnaryBackwardFuncBase {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        shared_ptr<TensorStorage> out = m_saved_tensors[0];
        shared_ptr<TensorStorage> res = mul(out, out_grad);
        m_input_nodes[0]->acc_grad(res);
    }
};

template<typename DT>
struct UnaryBackwardFuncImpl<DT, LogOp<DT>>: UnaryBackwardFuncBase {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        shared_ptr<TensorStorage> inp = m_saved_tensors[0];
        shared_ptr<TensorStorage> res = mul(reciprocal(inp), out_grad);
        m_input_nodes[0]->acc_grad(res);
    }
};

template<typename DT>
struct UnaryBackwardFuncImpl<DT, SigmoidOp<DT>>: UnaryBackwardFuncBase {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        shared_ptr<TensorStorage> inp = m_saved_tensors[0];
        shared_ptr<TensorStorage> res = mul(mul(sigmoid(inp), sigmoid(neg(inp))), out_grad);
        m_input_nodes[0]->acc_grad(res);
    }
};

template<typename DT>
struct UnaryBackwardFuncImpl<DT, NegOp<DT>>: UnaryBackwardFuncBase {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        shared_ptr<TensorStorage> res = neg(out_grad);
        m_input_nodes[0]->acc_grad(res);
    }
};

template<typename DT>
struct UnaryBackwardFuncImpl<DT, CopyOp<DT>>: UnaryBackwardFuncBase {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        m_input_nodes[0]->acc_grad(out_grad);
    }
};

template<typename DT>
struct UnaryBackwardFuncImpl<DT, ReciprocalOp<DT>>: UnaryBackwardFuncBase {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        shared_ptr<TensorStorage> inp = m_saved_tensors[0];
        shared_ptr<TensorStorage> res = neg(reciprocal(mul(inp, inp)));
        m_input_nodes[0]->acc_grad(res);
    }
};

template<typename DT>
struct UnaryBackwardFuncImpl<DT, AsFloat32Op<DT>>: UnaryBackwardFuncBase {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        if (std::is_same<DT, Float32>::value) {
            shared_ptr<TensorStorage> res = out_grad;
            m_input_nodes[0]->acc_grad(res);
        }
    }
};

// template<typename DT, typename Opr>
// struct UnaryBackwardFuncImpl: UnaryBackwardFunc<UnaryBackwardFuncImpl<DT, Opr>> {
// };
//
//
// template<typename DT>
// struct UnaryBackwardFuncImpl<DT, ReluOp<DT>>: UnaryBackwardFunc<UnaryBackwardFuncImpl<DT, ReluOp<DT>>> {
//     void backward_func(shared_ptr<GraphNode> out_node) override {
//         shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
//         shared_ptr<TensorStorage> inp = m_saved_tensors[0];
//         // shared_ptr<TensorStorage> res = mul(greater_then(inp, Tensor::get_const(0.).storage()), out_grad);
//         // m_input_nodes[0]->acc_grad(res);
//     }
// };
//
// template<typename DT>
// struct UnaryBackwardFuncImpl<DT, ExpOp<DT>>: UnaryBackwardFunc<UnaryBackwardFuncImpl<DT, ExpOp<DT>>> {
//     void backward_func(shared_ptr<GraphNode> out_node) override {
//         // shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
//         // shared_ptr<TensorStorage> out = m_saved_tensors[0];
//         // shared_ptr<TensorStorage> res = mul(out, out_grad);
//         // m_input_nodes[0]->acc_grad(res);
//     }
// };
//
// template<typename DT>
// struct UnaryBackwardFuncImpl<DT, LogOp<DT>>: UnaryBackwardFunc<UnaryBackwardFuncImpl<DT, LogOp<DT>>> {
//     void backward_func(shared_ptr<GraphNode> out_node) override {
//         // shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
//         // shared_ptr<TensorStorage> inp = m_saved_tensors[0];
//         // shared_ptr<TensorStorage> res = mul(reciprocal(inp), out_grad);
//         // m_input_nodes[0]->acc_grad(res);
//     }
// };
//
// template<typename DT>
// struct UnaryBackwardFuncImpl<DT, SigmoidOp<DT>>: UnaryBackwardFunc<UnaryBackwardFuncImpl<DT, SigmoidOp<DT>>> {
//     void backward_func(shared_ptr<GraphNode> out_node) override {
//         // shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
//         // shared_ptr<TensorStorage> inp = m_saved_tensors[0];
//         // shared_ptr<TensorStorage> res = mul(mul(sigmoid(inp), sigmoid(neg(inp))), out_grad);
//         // m_input_nodes[0]->acc_grad(res);
//     }
// };
//
// template<typename DT>
// struct UnaryBackwardFuncImpl<DT, NegOp<DT>>: UnaryBackwardFunc<UnaryBackwardFuncImpl<DT, NegOp<DT>>> {
//     void backward_func(shared_ptr<GraphNode> out_node) override {
//         // shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
//         // shared_ptr<TensorStorage> res = neg(out_grad);
//         // m_input_nodes[0]->acc_grad(res);
//     }
// };
//
// template<typename DT>
// struct UnaryBackwardFuncImpl<DT, CopyOp<DT>>: UnaryBackwardFunc<UnaryBackwardFuncImpl<DT, CopyOp<DT>>> {
//     void backward_func(shared_ptr<GraphNode> out_node) override {
//         // shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
//         // m_input_nodes[0]->acc_grad(out_grad);
//     }
// };
//
// template<typename DT>
// struct UnaryBackwardFuncImpl<DT, ReciprocalOp<DT>>: UnaryBackwardFunc<UnaryBackwardFuncImpl<DT, ReciprocalOp<DT>>> {
//     void backward_func(shared_ptr<GraphNode> out_node) override {
//         // shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
//         // shared_ptr<TensorStorage> inp = m_saved_tensors[0];
//         // shared_ptr<TensorStorage> res = neg(reciprocal(mul(inp, inp)));
//         // m_input_nodes[0]->acc_grad(res);
//     }
// };


template<typename Op>
Tensor unary_op(Tensor& x) {
    using DType = typename Op::DType;
    Tensor res;
    if(std::is_same<Op, CopyOp<DType>>::value) {
        res = Tensor(copy_op<typename Op::RT>(x.storage()));
    } else {
        res = Tensor(unary_op<Op>(x.storage()));
    }

    // bool do not support grad
    if(std::is_same<Op, AsBoolOp<DType>>::value || std::is_same<DType, Bool>::value) {
        return res;
    }

    if(x.need_grad()) {
        shared_ptr<GraphNode> out_node = res.graph_node();
        shared_ptr<GraphNode> x_node = x.graph_node();
        shared_ptr<BackwardFunc> func;

        if(std::is_same<Op, ReluOp<DType>>::value ||
                std::is_same<Op, LogOp<DType>>::value ||
                std::is_same<Op, SigmoidOp<DType>>::value ||
                std::is_same<Op, ReciprocalOp<DType>>::value) {
            func = UnaryBackwardFuncBase::make<Op>(x_node);
            func->m_saved_tensors.push_back(x.storage());
        } else if(std::is_same<Op, ExpOp<DType>>::value) {
            func = UnaryBackwardFuncBase::make<Op>(x_node);
            func->m_saved_tensors.push_back(res.storage());
        } else {
            func = UnaryBackwardFuncBase::make<Op>(x_node);
        }

        out_node->set_backward_func(func);
    }
    return res;
}

template<template<typename> typename Op>
Tensor dispatch_unary_op(Tensor& x) {
    switch(x.dtype()) {
        case DataType::Float32:
            return unary_op<Op<Float32>>(x);
        case DataType::UInt64:
            return unary_op<Op<UInt64>>(x);
        case DataType::Bool:
            return unary_op<Op<Bool>>(x);
        default:
            throw "Excepton";
    }
}

// *****************************************************************************
namespace opr {
Tensor add(Tensor& x, Tensor& y) {
    return dispatch_binary_op<AddOp>(x, y);
}

Tensor sub(Tensor& x, Tensor& y) {
    return dispatch_binary_op<SubOp>(x, y);
}

Tensor mul(Tensor& x, Tensor& y) {
    return dispatch_binary_op<MulOp>(x, y);
}

Tensor div(Tensor& x, Tensor& y) {
    return dispatch_binary_op<DivOp>(x, y);
}

Tensor equal(Tensor& x, Tensor& y) {
    return dispatch_binary_op<BooleanEqualOp>(x, y);
}

Tensor less_then(Tensor& x, Tensor& y) {
    return dispatch_binary_op<BooleanLessThenOp>(x, y);
}

Tensor less_equal(Tensor& x, Tensor& y) {
    return dispatch_binary_op<BooleanLessEqualOp>(x, y);
}

Tensor greater_then(Tensor& x, Tensor& y) {
    return dispatch_binary_op<BooleanGreaterThenOp>(x, y);
}

Tensor greater_equal(Tensor& x, Tensor& y) {
    return dispatch_binary_op<BooleanGreaterEqualOp>(x, y);
}

Tensor as_float32(Tensor& x) {
    return dispatch_unary_op<AsFloat32Op>(x);
}

Tensor as_bool(Tensor& x) {
    return dispatch_unary_op<AsBoolOp>(x);
}

Tensor relu(Tensor& x) {
    // FIXME:
    return dispatch_unary_op<ReluOp>(x);
}

Tensor log(Tensor& x) {
    // FIXME:
    return unary_op<LogOp<Float32>>(x);
}

Tensor exp(Tensor& x) {
    return unary_op<ExpOp<Float32>>(x);
}

Tensor sigmoid(Tensor& x) {
    // FIXME:
    return unary_op<SigmoidOp<Float32>>(x);
}

Tensor copy(Tensor& x) {
    return dispatch_unary_op<CopyOp>(x);
}

namespace intl {

}

}
