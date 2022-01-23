#include <cassert>
#include "tensor.h"
#include "elementwise.h"
#include "opr_utils.h"

using namespace std;

struct BinaryOpBackwarFunc;

// *****************************************************************************
enum class BinaryOpMode{
    UNDEFINED=0,
    ADD,
    SUB,
    MUL,
    DIV,
    EQ,
    LT,
    LE,
    GT,
    GE
};


enum class UnaryOpMode {
    UNDEFINED=0,
    RELU,
    EXP,
    LOG,
    SIGMOID,
    NEG,
    COPY,
    RECIPROCAL,
};


// *****************************************************************************
__global__ void kernel_binary_op(float *out, float *a, float *b, size_t n, BinaryOpMode mode) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n) {
        switch(mode) {
        case BinaryOpMode::ADD:
            out[idx] = a[idx] + b[idx];
            break;
        case BinaryOpMode::SUB:
            out[idx] = a[idx] - b[idx];
            break;
        case BinaryOpMode::MUL:
            out[idx] = a[idx] * b[idx];
            break;
        case BinaryOpMode::DIV:
            out[idx] = a[idx] / b[idx];
            break;
        case BinaryOpMode::EQ:
            out[idx] = a[idx] == b[idx] ? 1.0 : 0.0;
            break;
        case BinaryOpMode::LT:
            out[idx] = a[idx] < b[idx] ? 1.0 : 0.0;
            break;
        case BinaryOpMode::LE:
            out[idx] = a[idx] <= b[idx] ? 1.0 : 0.0;
            break;
        case BinaryOpMode::GT:
            out[idx] = a[idx] > b[idx] ? 1.0 : 0.0;
            break;
        case BinaryOpMode::GE:
            out[idx] = a[idx] >= b[idx] ? 1.0 : 0.0;
            break;
        default:
            break;
        }
    }
}


__global__ void kernel_binary_op(float *out, TensorFormat *out_format, 
        float *a, TensorFormat* a_format, 
        float *b, TensorFormat* b_format,
        size_t n, BinaryOpMode mode) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    int indices[TensorFormat::MAX_DIM];

    for(int idx0=idx, i=out_format->dim-1; i>=0; i--) {
        indices[i] = idx0 % out_format->shape[i];
        idx0 = idx0 / out_format->shape[i];
    }
    
    size_t idx_a = 0, idx_b = 0;
    for(int i=0; i<out_format->dim; i++) {
        size_t sa = a_format->shape[i] == 1 ? 0: (a_format->strides[i] / sizeof(float));
        size_t sb = b_format->shape[i] == 1 ? 0: (b_format->strides[i] / sizeof(float));
        idx_a  += sa * indices[i];
        idx_b  += sb * indices[i];
    }

    if(idx < n) {
        switch(mode) {
        case BinaryOpMode::ADD:
            out[idx] = a[idx_a] + b[idx_b];
            break;
        case BinaryOpMode::SUB:
            out[idx] = a[idx_a] - b[idx_b];
            break;
        case BinaryOpMode::MUL:
            out[idx] = a[idx_a] * b[idx_b];
            break;
        case BinaryOpMode::DIV:
            out[idx] = a[idx_a] / b[idx_b];
            break;
        case BinaryOpMode::EQ:
            out[idx] = a[idx_a] == b[idx_b] ? 1.0 : 0.0;
            break;
        case BinaryOpMode::LT:
            out[idx] = a[idx_a] < b[idx_b] ? 1.0 : 0.0;
            break;
        case BinaryOpMode::LE:
            out[idx] = a[idx_a] <= b[idx_b] ? 1.0 : 0.0;
            break;
        case BinaryOpMode::GT:
            out[idx] = a[idx_a] > b[idx_b] ? 1.0 : 0.0;
            break;
        case BinaryOpMode::GE:
            out[idx] = a[idx_a] >= b[idx_b] ? 1.0 : 0.0;
            break;
        default:
            break;
        }
    }
}


__global__ void kernel_unary_op(float *out, float *in, size_t n, UnaryOpMode mode) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n) {
        switch(mode) {
        case UnaryOpMode::RELU:
            out[idx] = in[idx] > 0.0f ? in[idx]: 0.0f;
            break;
        case UnaryOpMode::EXP:
            out[idx] = expf(in[idx]);
            break;
        case UnaryOpMode::LOG:
            out[idx] = logf(in[idx]);
            break;
        case UnaryOpMode::SIGMOID:
            out[idx] = 1.f / (1 + expf(in[idx]));
            break;
        case UnaryOpMode::NEG:
            out[idx] = - in[idx];
            break;
        case UnaryOpMode::RECIPROCAL:
            out[idx] = 1.f / in[idx];
            break;
        default:
            break;
        }
    }
}

// *****************************************************************************
shared_ptr<TensorStorage> binary_op(BinaryOpMode mode, shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    bool same_layout = true;

    same_layout = same_layout && (x->size() == y->size());
    same_layout = same_layout && (x->dim() == y->dim());
    for(size_t i=0; i<x->dim(); i++)
        same_layout = same_layout && (x->m_shape[i] == y->m_shape[i]);

    float *res;
    if(same_layout) {
        cudaMalloc(&res, sizeof(float) * x->size());

        int block_size = 128;
        int n_block = (x->size() + block_size - 1) / block_size;
        kernel_binary_op<<<n_block, block_size>>>(res, x->m_data, y->m_data, x->size(), mode);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("cuda error %s\n", cudaGetErrorString(err));
        }
        return make_shared<TensorStorage>(res, x->size(), x->m_shape, x->m_strides);
    } else {
        vector<size_t> out_shape;
        vector<size_t> out_strides;
        TensorFormat *x_format, *y_format, *out_format;
        size_t res_size;

        if(x->dim() == y->dim()) {
            out_shape.resize(x->dim());
            out_strides.resize(x->dim());
            size_t s = 1;
            for(int i=x->dim() - 1;i >= 0; i--) {
                out_shape[i] = max(x->m_shape[i], y->m_shape[i]);
                out_strides[i] = s * sizeof(float);
                s *= out_shape[i];
            }
            res_size = s;
            x_format = TensorFormat::make_cuda_tensor_format(x->m_shape, x->m_strides);
            y_format = TensorFormat::make_cuda_tensor_format(y->m_shape, y->m_strides);
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
                out_shape[i] = y->m_shape[i];
                out_strides[i] = s * sizeof(float);
                s *= out_shape[i];
                x_shape[i] = 1;
                x_strides[i] = 1;
            }
            res_size = s;
            x_format = TensorFormat::make_cuda_tensor_format(x_shape, x_strides);
            y_format = TensorFormat::make_cuda_tensor_format(y->m_shape, y->m_strides);
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
                out_shape[i] = x->m_shape[i];
                out_strides[i] = s * sizeof(float);
                s *= out_shape[i];
                y_shape[i] = 1;
                y_strides[i] = 1;
            }
            res_size = s;
            x_format = TensorFormat::make_cuda_tensor_format(x->m_shape, x->m_strides);
            y_format = TensorFormat::make_cuda_tensor_format(y_shape, y_strides);
            out_format = TensorFormat::make_cuda_tensor_format(out_shape, out_strides);
        } else {
            throw "Error";
        }
        
        cudaMalloc(&res, sizeof(float) * res_size);

        int block_size = 128;
        int n_block = (res_size + block_size - 1) / block_size;

        kernel_binary_op<<<n_block, block_size>>>(res, out_format,
                x->m_data, x_format, 
                y->m_data, y_format, 
                res_size, mode);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("cuda error %s\n", cudaGetErrorString(err));
        }
        return make_shared<TensorStorage>(res, res_size, out_shape, out_strides);
    }
}


shared_ptr<TensorStorage> unary_op(UnaryOpMode mode, shared_ptr<TensorStorage> x) {
    float *res;
    cudaMalloc(&res, sizeof(float) * x->size());

    int block_size = 128;
    int n_block = (x->size() + block_size - 1) / block_size;
    kernel_unary_op<<<n_block, block_size>>>(res, x->m_data, x->size(), mode);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cuda error %s\n", cudaGetErrorString(err));
    }
    return make_shared<TensorStorage>(res, x->size(), x->m_shape, x->m_strides);
}


shared_ptr<TensorStorage> copy_op(shared_ptr<TensorStorage> x) {
    float *res;
    // FIXME: multiply sizeof(float)??
    cudaMalloc(&res, sizeof(float) * x->size());

    cudaMemcpy(res, x->m_data, sizeof(float) * x->size(), cudaMemcpyDeviceToDevice);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cuda error %s\n", cudaGetErrorString(err));
    }
    return make_shared<TensorStorage>(res, x->size(), x->m_shape, x->m_strides);
}


// *****************************************************************************

shared_ptr<TensorStorage> add(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return binary_op(BinaryOpMode::ADD, x, y);
}

shared_ptr<TensorStorage> sub(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return binary_op(BinaryOpMode::SUB, x, y);
}

shared_ptr<TensorStorage> mul(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return binary_op(BinaryOpMode::MUL, x, y);
}

shared_ptr<TensorStorage> div(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return binary_op(BinaryOpMode::DIV, x, y);
}

shared_ptr<TensorStorage> equal(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return binary_op(BinaryOpMode::EQ, x, y);
}

shared_ptr<TensorStorage> less_then(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return binary_op(BinaryOpMode::LT, x, y);
}

shared_ptr<TensorStorage> less_equal(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return binary_op(BinaryOpMode::LE, x, y);
}

shared_ptr<TensorStorage> greater_then(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return binary_op(BinaryOpMode::GT, x, y);
}

shared_ptr<TensorStorage> greater_equal(shared_ptr<TensorStorage> x, shared_ptr<TensorStorage> y) {
    return binary_op(BinaryOpMode::GE, x, y);
}

shared_ptr<TensorStorage> relu(shared_ptr<TensorStorage> x) {
    return unary_op(UnaryOpMode::RELU, x);
}

shared_ptr<TensorStorage> exp(shared_ptr<TensorStorage> x) {
    return unary_op(UnaryOpMode::EXP, x);
}

shared_ptr<TensorStorage> log(shared_ptr<TensorStorage> x) {
    return unary_op(UnaryOpMode::LOG, x);
}

shared_ptr<TensorStorage> sigmoid(shared_ptr<TensorStorage> x) {
    return unary_op(UnaryOpMode::SIGMOID, x);
}

shared_ptr<TensorStorage> neg(shared_ptr<TensorStorage> x) {
    return unary_op(UnaryOpMode::NEG, x);
}

shared_ptr<TensorStorage> copy(shared_ptr<TensorStorage> x) {
    return copy_op(x);
}

shared_ptr<TensorStorage> reciprocal(shared_ptr<TensorStorage> x) {
    return unary_op(UnaryOpMode::RECIPROCAL, x);
}

// *****************************************************************************
struct BinaryOpBackwarFunc: BackwardFunc {
    BinaryOpMode m_mode;
    static std::shared_ptr<BackwardFunc> make(shared_ptr<GraphNode> x, shared_ptr<GraphNode> y, BinaryOpMode mode){
        shared_ptr<BackwardFunc> func = make_shared<BinaryOpBackwarFunc>(mode);
        func->m_input_nodes.push_back(x);
        func->m_input_nodes.push_back(y);
        return func;
    }

    void backward_func(shared_ptr<GraphNode> out_node) override {
        // TODO: check shape
        switch(m_mode) {
            case BinaryOpMode::ADD:
                m_input_nodes[0]->acc_grad(out_node->m_grad_storage);
                m_input_nodes[1]->acc_grad(out_node->m_grad_storage);
                break;
            case BinaryOpMode::SUB:
                m_input_nodes[0]->acc_grad(out_node->m_grad_storage);
                m_input_nodes[1]->acc_grad(
                        unary_op(UnaryOpMode::NEG, out_node->m_grad_storage));
                break;
            case BinaryOpMode::MUL:
                m_input_nodes[0]->acc_grad(
                        binary_op(BinaryOpMode::MUL, 
                            out_node->m_grad_storage,
                            m_saved_tensors[1]));
                m_input_nodes[1]->acc_grad(
                        binary_op(BinaryOpMode::MUL,
                            out_node->m_grad_storage,
                            m_saved_tensors[0]));
                break;
            case BinaryOpMode::DIV:
                m_input_nodes[0]->acc_grad(
                        div(out_node->m_grad_storage, m_saved_tensors[1]));
                m_input_nodes[1]->acc_grad(
                        neg(div(mul(out_node->m_grad_storage, m_saved_tensors[0]), 
                                mul(m_saved_tensors[1], m_saved_tensors[1]))));
                break;
            default:
                break;
        }
    }

    BinaryOpBackwarFunc(BinaryOpMode mode): m_mode(mode) {}
};


Tensor binary_op(BinaryOpMode mode, Tensor& x, Tensor& y) {
    Tensor res = Tensor(binary_op(mode, x.m_storage, y.m_storage));
    if(x.m_need_grad || y.m_need_grad || x.m_require_grad || y.m_require_grad) {
        shared_ptr<GraphNode> x_node = x.graph_node();
        shared_ptr<GraphNode> y_node = y.graph_node();
        shared_ptr<GraphNode> out_node = res.graph_node();
        shared_ptr<BackwardFunc> func = BinaryOpBackwarFunc::make(x_node, y_node, mode);
        switch(mode) {
        case BinaryOpMode::MUL:
        case BinaryOpMode::DIV:
            func->m_saved_tensors.push_back(x.m_storage);
            func->m_saved_tensors.push_back(y.m_storage);
            break;
        default:
            break;
        }
        out_node->set_backward_func(func);
        out_node->m_need_grad = true;
        res.m_need_grad = true;
    }
    return res;
}

// *****************************************************************************
template<typename T>
struct UnaryBackwardFunc: BackwardFunc {
    static std::shared_ptr<BackwardFunc> make(shared_ptr<GraphNode> x){
        shared_ptr<BackwardFunc> func = make_shared<T>();
        func->m_input_nodes.push_back(x);
        return func;
    }
    void backward_func(shared_ptr<GraphNode> out_node) {
    }
};

struct ReLUBackwarFunc: UnaryBackwardFunc<ReLUBackwarFunc> {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->m_grad_storage;
        shared_ptr<TensorStorage> inp = m_saved_tensors[0];
        shared_ptr<TensorStorage> res = mul(greater_then(inp, Tensor::get_const(0.).m_storage), out_grad);
        m_input_nodes[0]->acc_grad(res);
    }
};

struct ExpBackwarFunc: UnaryBackwardFunc<ExpBackwarFunc> {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->m_grad_storage;
        shared_ptr<TensorStorage> out = m_saved_tensors[0];
        shared_ptr<TensorStorage> res = mul(out, out_grad);
        m_input_nodes[0]->acc_grad(res);
    }
};

struct LogBackwarFunc: UnaryBackwardFunc<LogBackwarFunc> {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->m_grad_storage;
        shared_ptr<TensorStorage> inp = m_saved_tensors[0];
        shared_ptr<TensorStorage> res = mul(reciprocal(inp), out_grad);
        m_input_nodes[0]->acc_grad(res);
    }
};

struct SigmoidBackwarFunc: UnaryBackwardFunc<SigmoidBackwarFunc> {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->m_grad_storage;
        shared_ptr<TensorStorage> inp = m_saved_tensors[0];
        shared_ptr<TensorStorage> res = mul(mul(sigmoid(inp), sigmoid(neg(inp))), out_grad);
        m_input_nodes[0]->acc_grad(res);
    }
};

struct NegBackwarFunc: UnaryBackwardFunc<NegBackwarFunc> {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->m_grad_storage;
        shared_ptr<TensorStorage> res = neg(out_grad);
        m_input_nodes[0]->acc_grad(res);
    }
};

struct CopyBackwarFunc: UnaryBackwardFunc<CopyBackwarFunc> {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->m_grad_storage;
        m_input_nodes[0]->acc_grad(out_grad);
    }
};

struct ReciprocalBackwarFunc: UnaryBackwardFunc<ReciprocalBackwarFunc> {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->m_grad_storage;
        shared_ptr<TensorStorage> inp = m_saved_tensors[0];
        shared_ptr<TensorStorage> res = neg(reciprocal(mul(inp, inp)));
        m_input_nodes[0]->acc_grad(res);
    }
};

Tensor unary_op(UnaryOpMode mode, Tensor& x) {
    if(mode == UnaryOpMode::COPY)
        return copy_op(x.m_storage);
    Tensor res = Tensor(unary_op(mode, x.m_storage));
    if(x.m_require_grad || x.m_need_grad) {
        shared_ptr<GraphNode> out_node = res.graph_node();
        shared_ptr<GraphNode> x_node = x.graph_node();
        shared_ptr<BackwardFunc> func;
        switch(mode) {
        case UnaryOpMode::RELU:
            func = ReLUBackwarFunc::make(x_node);
            func->m_saved_tensors.push_back(x.m_storage);
            break;
        case UnaryOpMode::EXP:
            func = ExpBackwarFunc::make(x_node);
            func->m_saved_tensors.push_back(res.m_storage);
            break;
        case UnaryOpMode::LOG:
            func = LogBackwarFunc::make(x_node);
            func->m_saved_tensors.push_back(x.m_storage);
            break;
        case UnaryOpMode::SIGMOID:
            func = SigmoidBackwarFunc::make(x_node);
            func->m_saved_tensors.push_back(x.m_storage);
            break;
        case UnaryOpMode::NEG:
            func = NegBackwarFunc::make(x_node);
            break;
        case UnaryOpMode::COPY:
            func = CopyBackwarFunc::make(x_node);
            break;
        case UnaryOpMode::RECIPROCAL:
            func = ReciprocalBackwarFunc::make(x_node);
            func->m_saved_tensors.push_back(x.m_storage);
            break;
        default:
            break;
        }
        out_node->set_backward_func(func);
        out_node->m_need_grad = true;
        res.m_need_grad = true;
    }
    return res;
}


// *****************************************************************************
namespace opr {
Tensor add(Tensor& x, Tensor& y) {
    return binary_op(BinaryOpMode::ADD, x, y);
}

Tensor sub(Tensor& x, Tensor& y) {
    return binary_op(BinaryOpMode::SUB, x, y);
}

Tensor mul(Tensor& x, Tensor& y) {
    return binary_op(BinaryOpMode::MUL, x, y);
}

Tensor div(Tensor& x, Tensor& y) {
    return binary_op(BinaryOpMode::DIV, x, y);
}

Tensor relu(Tensor& x) {
    return unary_op(UnaryOpMode::RELU, x);
}

Tensor log(Tensor& x) {
    return unary_op(UnaryOpMode::LOG, x);
}

Tensor exp(Tensor& x) {
    return unary_op(UnaryOpMode::EXP, x);
}

Tensor sigmoid(Tensor& x) {
    return unary_op(UnaryOpMode::SIGMOID, x);
}

Tensor copy(Tensor& x) {
    return unary_op(UnaryOpMode::COPY, x);
}

}
