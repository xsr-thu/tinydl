#include <cassert>
#include "tensor.h"
#include "elementwise.h"
#include "reduction.h"
#include "opr_utils.h"

using namespace std;

struct BinaryOpBackwarFuncBase;

// *****************************************************************************
struct ReluOp {
    static __device__ __forceinline__ float apply(float x) {
        return x > 0.0f ? x: 0.0f;
    }
};

struct ExpOp {
    static __device__ __forceinline__ float apply(float x) {
        return expf(x);
    }
};

struct LogOp {
    static __device__ __forceinline__ float apply(float x) {
        return logf(x);
    }
};

struct SigmoidOp {
    static __device__ __forceinline__ float apply(float x) {
        return 1.f / (1.f + expf(x));
    }
};

struct NegOp {
    static __device__ __forceinline__ float apply(float x) {
        return -x;
    }
};

struct CopyOp {
    static __device__ __forceinline__ float apply(float x) {
        return x;
    }
};

struct ReciprocalOp {
    static __device__ __forceinline__ float apply(float x) {
        return 1.f / x;
    }
};


struct AddOp {
    static __device__ __forceinline__ float apply(float lhs, float rhs) {
        return lhs + rhs;
    }
};

struct SubOp {
    static __device__ __forceinline__ float apply(float lhs, float rhs) {
        return lhs - rhs;
    }
};

struct MulOp {
    static __device__ __forceinline__ float apply(float lhs, float rhs) {
        return lhs * rhs;
    }
};

struct DivOp {
    static __device__ __forceinline__ float apply(float lhs, float rhs) {
        return lhs / rhs;
    }
};

struct EqualOp {
    static __device__ __forceinline__ float apply(float lhs, float rhs) {
        return lhs == rhs ? 1.0: 0.0;
    }
};

struct LessThenOp {
    static __device__ __forceinline__ float apply(float lhs, float rhs) {
        return lhs < rhs ? 1.0: 0.0;
    }
};

struct LessEqualOp {
    static __device__ __forceinline__ float apply(float lhs, float rhs) {
        return lhs <= rhs ? 1.0: 0.0;
    }
};

struct GreaterThenOp {
    static __device__ __forceinline__ float apply(float lhs, float rhs) {
        return lhs > rhs ? 1.0: 0.0;
    }
};

struct GreaterEqualOp {
    static __device__ __forceinline__ float apply(float lhs, float rhs) {
        return lhs >= rhs ? 1.0: 0.0;
    }
};

// *****************************************************************************
template<typename Op>
__global__ void kernel_binary_op(float *out, float *a, float *b, size_t n) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n) {
        out[idx] = Op::apply(a[idx], b[idx]);
    }
}


template<typename Op>
__global__ void kernel_binary_op(float *out, TensorFormat *out_format, 
        float *a, TensorFormat* a_format, 
        float *b, TensorFormat* b_format,
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
__global__ void kernel_unary_op(float *out, float *in, size_t n) {
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

    float *res;
    if(same_layout) {
        cudaMalloc(&res, sizeof(float) * x->size());

        int block_size = 128;
        int n_block = (x->size() + block_size - 1) / block_size;
        kernel_binary_op<Op><<<n_block, block_size>>>(res, x->data(), y->data(), x->size());
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("cuda error %s\n", cudaGetErrorString(err));
        }
        return make_shared<TensorStorage>(res, x->size(), x->shape(), x->strides());
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
        
        cudaMalloc(&res, sizeof(float) * res_size);

        int block_size = 128;
        int n_block = (res_size + block_size - 1) / block_size;

        kernel_binary_op<Op><<<n_block, block_size>>>(res, out_format.get(),
                x->data(), x_format.get(),
                y->data(), y_format.get(),
                res_size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("cuda error %s\n", cudaGetErrorString(err));
        }
        return make_shared<TensorStorage>(res, res_size, out_shape, out_strides);
    }
}


template<typename Op>
shared_ptr<TensorStorage> unary_op(shared_ptr<TensorStorage> x) {
    float *res;
    cudaMalloc(&res, sizeof(float) * x->size());

    int block_size = 128;
    int n_block = (x->size() + block_size - 1) / block_size;
    kernel_unary_op<Op><<<n_block, block_size>>>(res, x->data(), x->size());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cuda error %s\n", cudaGetErrorString(err));
    }
    return make_shared<TensorStorage>(res, x->size(), x->shape(), x->strides());
}


shared_ptr<TensorStorage> copy_op(shared_ptr<TensorStorage> x) {
    float *res;
    // FIXME: multiply sizeof(float)??
    cudaMalloc(&res, sizeof(float) * x->size());

    cudaMemcpy(res, x->data(), sizeof(float) * x->size(), cudaMemcpyDeviceToDevice);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cuda error %s\n", cudaGetErrorString(err));
    }
    return make_shared<TensorStorage>(res, x->size(), x->shape(), x->strides());
}


// *****************************************************************************

shared_ptr<TensorStorage> add(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return binary_op<AddOp>(x, y);
}

shared_ptr<TensorStorage> sub(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return binary_op<SubOp>(x, y);
}

shared_ptr<TensorStorage> mul(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return binary_op<MulOp>(x, y);
}

shared_ptr<TensorStorage> div(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return binary_op<DivOp>(x, y);
}

shared_ptr<TensorStorage> equal(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return binary_op<EqualOp>(x, y);
}

shared_ptr<TensorStorage> less_then(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return binary_op<LessThenOp>(x, y);
}

shared_ptr<TensorStorage> less_equal(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return binary_op<LessEqualOp>(x, y);
}

shared_ptr<TensorStorage> greater_then(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return binary_op<GreaterThenOp>(x, y);
}

shared_ptr<TensorStorage> greater_equal(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return binary_op<GreaterEqualOp>(x, y);
}

shared_ptr<TensorStorage> relu(shared_ptr<TensorStorage> x) {
    return unary_op<ReluOp>(x);
}

shared_ptr<TensorStorage> exp(shared_ptr<TensorStorage> x) {
    return unary_op<ExpOp>(x);
}

shared_ptr<TensorStorage> log(shared_ptr<TensorStorage> x) {
    return unary_op<LogOp>(x);
}

shared_ptr<TensorStorage> sigmoid(shared_ptr<TensorStorage> x) {
    return unary_op<SigmoidOp>(x);
}

shared_ptr<TensorStorage> neg(shared_ptr<TensorStorage> x) {
    return unary_op<NegOp>(x);
}

shared_ptr<TensorStorage> copy(shared_ptr<TensorStorage> x) {
    return copy_op(x);
}

shared_ptr<TensorStorage> reciprocal(shared_ptr<TensorStorage> x) {
    return unary_op<ReciprocalOp>(x);
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

template<typename Op>
struct BinaryOpBackwarFunc: BinaryOpBackwarFuncBase {
    BinaryOpBackwarFunc(const vector<size_t> &x_shape, const vector<size_t> &y_shape)
        :BinaryOpBackwarFuncBase(x_shape, y_shape) {};

    void backward_func(shared_ptr<GraphNode> out_node) override {
        // FIXME: not implemented for GT GE etc.
    }
};


template<>
struct BinaryOpBackwarFunc<AddOp>: BinaryOpBackwarFuncBase {
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


template<>
struct BinaryOpBackwarFunc<SubOp>: BinaryOpBackwarFuncBase {
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
                        unary_op<NegOp>(out_node->grad_storage()),
                        y_reduce_dims, true));
        } else {
            m_input_nodes[1]->acc_grad(
                    unary_op<NegOp>(out_node->grad_storage()));
        }

    }
};

template<>
struct BinaryOpBackwarFunc<MulOp>: BinaryOpBackwarFuncBase {
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
                        binary_op<MulOp>(out_node->grad_storage(), m_saved_tensors[1]),
                        x_reduce_dims, true));
        } else {
            m_input_nodes[0]->acc_grad(
                    binary_op<MulOp>(out_node->grad_storage(), m_saved_tensors[1]));
        }

        if(y_reduce_dims.size()) {
            // fprintf(stderr, " backprop with reduction y n=%zu first=%zu\n", y_reduce_dims.size(), y_reduce_dims[0]);
            m_input_nodes[1]->acc_grad(
                    opr::intl::reduce_sum(
                        binary_op<MulOp>(out_node->grad_storage(), m_saved_tensors[0]),
                        y_reduce_dims, true));
        } else {
            m_input_nodes[1]->acc_grad(
                    binary_op<MulOp>(out_node->grad_storage(), m_saved_tensors[0]));
        }

    }
};

template<>
struct BinaryOpBackwarFunc<DivOp>: BinaryOpBackwarFuncBase {
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
    shared_ptr<BackwardFunc> func = make_shared<BinaryOpBackwarFunc<Op>>(x_shape, y_shape);
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
        if(std::is_same<Op, MulOp>::value || std::is_same<Op, DivOp>::value) {
            func->m_saved_tensors.push_back(x.storage());
            func->m_saved_tensors.push_back(y.storage());
        }
        out_node->set_backward_func(func);
    }
    return res;
}

// *****************************************************************************

template<typename Op>
struct UnaryBackwardFuncImpl: UnaryBackwardFunc<UnaryBackwardFuncImpl<Op>> {
};


template<>
struct UnaryBackwardFuncImpl<ReluOp>: UnaryBackwardFunc<UnaryBackwardFuncImpl<ReluOp>> {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        shared_ptr<TensorStorage> inp = m_saved_tensors[0];
        shared_ptr<TensorStorage> res = mul(greater_then(inp, Tensor::get_const(0.).storage()), out_grad);
        m_input_nodes[0]->acc_grad(res);
    }
};

template<>
struct UnaryBackwardFuncImpl<ExpOp>: UnaryBackwardFunc<UnaryBackwardFuncImpl<ExpOp>> {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        shared_ptr<TensorStorage> out = m_saved_tensors[0];
        shared_ptr<TensorStorage> res = mul(out, out_grad);
        m_input_nodes[0]->acc_grad(res);
    }
};

template<>
struct UnaryBackwardFuncImpl<LogOp>: UnaryBackwardFunc<UnaryBackwardFuncImpl<LogOp>> {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        shared_ptr<TensorStorage> inp = m_saved_tensors[0];
        shared_ptr<TensorStorage> res = mul(reciprocal(inp), out_grad);
        m_input_nodes[0]->acc_grad(res);
    }
};

template<>
struct UnaryBackwardFuncImpl<SigmoidOp>: UnaryBackwardFunc<UnaryBackwardFuncImpl<SigmoidOp>> {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        shared_ptr<TensorStorage> inp = m_saved_tensors[0];
        shared_ptr<TensorStorage> res = mul(mul(sigmoid(inp), sigmoid(neg(inp))), out_grad);
        m_input_nodes[0]->acc_grad(res);
    }
};

template<>
struct UnaryBackwardFuncImpl<NegOp>: UnaryBackwardFunc<UnaryBackwardFuncImpl<NegOp>> {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        shared_ptr<TensorStorage> res = neg(out_grad);
        m_input_nodes[0]->acc_grad(res);
    }
};

template<>
struct UnaryBackwardFuncImpl<CopyOp>: UnaryBackwardFunc<UnaryBackwardFuncImpl<CopyOp>> {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        m_input_nodes[0]->acc_grad(out_grad);
    }
};

template<>
struct UnaryBackwardFuncImpl<ReciprocalOp>: UnaryBackwardFunc<UnaryBackwardFuncImpl<ReciprocalOp>> {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        shared_ptr<TensorStorage> inp = m_saved_tensors[0];
        shared_ptr<TensorStorage> res = neg(reciprocal(mul(inp, inp)));
        m_input_nodes[0]->acc_grad(res);
    }
};


template<typename Op>
Tensor unary_op(Tensor& x) {
    Tensor res;
    if(std::is_same<Op, CopyOp>::value) {
        res = Tensor(copy_op(x.storage()));
    } else {
        res = Tensor(unary_op<Op>(x.storage()));
    }

    if(x.need_grad()) {
        shared_ptr<GraphNode> out_node = res.graph_node();
        shared_ptr<GraphNode> x_node = x.graph_node();
        shared_ptr<BackwardFunc> func;

        if(std::is_same<Op, ReluOp>::value ||
                std::is_same<Op, LogOp>::value ||
                std::is_same<Op, SigmoidOp>::value ||
                std::is_same<Op, ReciprocalOp>::value) {
            func = UnaryBackwardFuncImpl<Op>::make(x_node);
            func->m_saved_tensors.push_back(x.storage());
        } else if(std::is_same<Op, ExpOp>::value) {
            func = UnaryBackwardFuncImpl<Op>::make(x_node);
            func->m_saved_tensors.push_back(res.storage());
        } else {
            func = UnaryBackwardFuncImpl<Op>::make(x_node);
        }

        out_node->set_backward_func(func);
    }
    return res;
}


// *****************************************************************************
namespace opr {
Tensor add(Tensor& x, Tensor& y) {
    return binary_op<AddOp>(x, y);
}

Tensor sub(Tensor& x, Tensor& y) {
    return binary_op<SubOp>(x, y);
}

Tensor mul(Tensor& x, Tensor& y) {
    return binary_op<MulOp>(x, y);
}

Tensor div(Tensor& x, Tensor& y) {
    return binary_op<DivOp>(x, y);
}

Tensor equal(Tensor& x, Tensor& y) {
    return binary_op<EqualOp>(x, y);
}

Tensor less_then(Tensor& x, Tensor& y) {
    return binary_op<LessThenOp>(x, y);
}

Tensor less_equal(Tensor& x, Tensor& y) {
    return binary_op<LessEqualOp>(x, y);
}

Tensor greater_then(Tensor& x, Tensor& y) {
    return binary_op<GreaterThenOp>(x, y);
}

Tensor greater_equal(Tensor& x, Tensor& y) {
    return binary_op<GreaterEqualOp>(x, y);
}

Tensor relu(Tensor& x) {
    return unary_op<ReluOp>(x);
}

Tensor log(Tensor& x) {
    return unary_op<LogOp>(x);
}

Tensor exp(Tensor& x) {
    return unary_op<ExpOp>(x);
}

Tensor sigmoid(Tensor& x) {
    return unary_op<SigmoidOp>(x);
}

Tensor copy(Tensor& x) {
    return unary_op<CopyOp>(x);
}

namespace intl {

}

}
