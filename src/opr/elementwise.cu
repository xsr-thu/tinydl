#include <cassert>
#include "tensor.h"
#include "elementwise.h"
#include "reduction.h"
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
        same_layout = same_layout && (x->shape()[i] == y->shape()[i]) && (x->strides()[i] == y->strides()[i]);

    float *res;
    if(same_layout) {
        cudaMalloc(&res, sizeof(float) * x->size());

        int block_size = 128;
        int n_block = (x->size() + block_size - 1) / block_size;
        kernel_binary_op<<<n_block, block_size>>>(res, x->data(), y->data(), x->size(), mode);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("cuda error %s\n", cudaGetErrorString(err));
        }
        return make_shared<TensorStorage>(res, x->size(), x->shape(), x->strides());
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
                out_shape[i] = max(x->shape()[i], y->shape()[i]);
                out_strides[i] = s * sizeof(float);
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
                out_strides[i] = s * sizeof(float);
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
                out_strides[i] = s * sizeof(float);
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

        kernel_binary_op<<<n_block, block_size>>>(res, out_format,
                x->data(), x_format, 
                y->data(), y_format, 
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
    kernel_unary_op<<<n_block, block_size>>>(res, x->data(), x->size(), mode);
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
    vector<size_t> m_x_shape;
    vector<size_t> m_y_shape;
    static std::shared_ptr<BackwardFunc> make(
            shared_ptr<GraphNode> x, const vector<size_t> &x_shape,
            shared_ptr<GraphNode> y, const vector<size_t> &y_shape,
            BinaryOpMode mode){
        shared_ptr<BackwardFunc> func = make_shared<BinaryOpBackwarFunc>(mode, x_shape, y_shape);
        func->m_input_nodes.push_back(x);
        func->m_input_nodes.push_back(y);
        return func;
    }

    void backward_func(shared_ptr<GraphNode> out_node) override {
        // TODO: check shape
        vector<size_t> x_reduce_dims;
        vector<size_t> y_reduce_dims;
        assert(out_node->dim() == m_x_shape.size());
        assert(out_node->dim() == m_y_shape.size());

        // fprintf(stderr, "=========================\n");
        for(int i=0; i<out_node->dim(); i++) {
            if(out_node->shape()[i] != m_x_shape[i]) {
                // fprintf(stderr, "dim %d %zu %zu\n", i, out_node->shape()[i], m_x_shape[i]);
                assert(m_x_shape[i] == 1);
                x_reduce_dims.push_back(i);
            }
            if(out_node->shape()[i] != m_y_shape[i]) {
                // fprintf(stderr, "dim %d %zu %zu\n", i, out_node->shape()[i], m_y_shape[i]);
                assert(m_y_shape[i] == 1);
                y_reduce_dims.push_back(i);
            }
        }
        // FIXME: if inputs do not need grad, does not backprop


        // fprintf(stderr, "  - X %s\n", to_string(m_x_shape).c_str());
        // fprintf(stderr, "  - Y %s\n", to_string(m_y_shape).c_str());

        switch(m_mode) {
            case BinaryOpMode::ADD:
                // m_input_nodes[0]->acc_grad(out_node->grad_storage());
                // m_input_nodes[1]->acc_grad(out_node->grad_storage());
                // fprintf(stderr, "   add\n");
                if(x_reduce_dims.size()) {
                    // fprintf(stderr, " backprop with reduction x n=%zu first=%zu\n", x_reduce_dims.size(), x_reduce_dims[0]);
                    m_input_nodes[0]->acc_grad(
                            opr::intl::reduce_sum(out_node->grad_storage(), x_reduce_dims, true));
                } else {
                    m_input_nodes[0]->acc_grad(out_node->grad_storage());
                }

                if(y_reduce_dims.size()) {
                    // fprintf(stderr, " backprop with reduction y n=%zu first=%zu\n", y_reduce_dims.size(), y_reduce_dims[0]);
                    m_input_nodes[1]->acc_grad(
                            opr::intl::reduce_sum(out_node->grad_storage(), y_reduce_dims, true));
                } else {
                    m_input_nodes[1]->acc_grad(out_node->grad_storage());
                }

                break;
            case BinaryOpMode::SUB:
                // fprintf(stderr, "   sub\n");
                if(x_reduce_dims.size()) {
                    // fprintf(stderr, " backprop with reduction x n=%zu first=%zu\n", x_reduce_dims.size(), x_reduce_dims[0]);
                    m_input_nodes[0]->acc_grad(
                            opr::intl::reduce_sum(out_node->grad_storage(), x_reduce_dims, true));
                } else {
                    m_input_nodes[0]->acc_grad(out_node->grad_storage());
                }

                if(y_reduce_dims.size()) {
                    // fprintf(stderr, " backprop with reduction y n=%zu first=%zu\n", y_reduce_dims.size(), y_reduce_dims[0]);
                    m_input_nodes[1]->acc_grad(
                            opr::intl::reduce_sum(
                                unary_op(UnaryOpMode::NEG, out_node->grad_storage()),
                                y_reduce_dims, true));
                } else {
                    m_input_nodes[1]->acc_grad(
                            unary_op(UnaryOpMode::NEG, out_node->grad_storage()));
                }

                break;
            case BinaryOpMode::MUL:
                // fprintf(stderr, "   mul\n");

                if(x_reduce_dims.size()) {
                    // fprintf(stderr, " backprop with reduction x n=%zu first=%zu\n", x_reduce_dims.size(), x_reduce_dims[0]);
                    m_input_nodes[0]->acc_grad(
                            opr::intl::reduce_sum(
                                binary_op(BinaryOpMode::MUL, out_node->grad_storage(), m_saved_tensors[1]),
                                x_reduce_dims, true));
                } else {
                    m_input_nodes[0]->acc_grad(
                            binary_op(BinaryOpMode::MUL,
                                out_node->grad_storage(),
                                m_saved_tensors[1]));
                }

                if(y_reduce_dims.size()) {
                    // fprintf(stderr, " backprop with reduction y n=%zu first=%zu\n", y_reduce_dims.size(), y_reduce_dims[0]);
                    m_input_nodes[1]->acc_grad(
                            opr::intl::reduce_sum(
                                binary_op(BinaryOpMode::MUL, out_node->grad_storage(), m_saved_tensors[0]),
                                y_reduce_dims, true));
                } else {
                    m_input_nodes[1]->acc_grad(
                            binary_op(BinaryOpMode::MUL,
                                out_node->grad_storage(),
                                m_saved_tensors[0]));
                }
                break;
            case BinaryOpMode::DIV:
                // fprintf(stderr, "   div\n");

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



                break;
            default:
                break;
        }
        // fprintf(stderr, "---------------\n");
    }

    BinaryOpBackwarFunc(BinaryOpMode mode, const vector<size_t> &x_shape, const vector<size_t> &y_shape)
        : m_mode(mode), m_x_shape(x_shape), m_y_shape(y_shape) {}
};


Tensor binary_op(BinaryOpMode mode, Tensor& x, Tensor& y) {
    Tensor res = Tensor(binary_op(mode, x.storage(), y.storage()));
    if(x.need_grad() || y.need_grad()) {
        shared_ptr<GraphNode> x_node = x.graph_node();
        shared_ptr<GraphNode> y_node = y.graph_node();
        shared_ptr<GraphNode> out_node = res.graph_node();
        shared_ptr<BackwardFunc> func = BinaryOpBackwarFunc::make(x_node, x.shape(), y_node, y.storage()->shape(), mode);
        switch(mode) {
        case BinaryOpMode::MUL:
        case BinaryOpMode::DIV:
            func->m_saved_tensors.push_back(x.storage());
            func->m_saved_tensors.push_back(y.storage());
            break;
        default:
            break;
        }
        out_node->set_backward_func(func);
    }
    return res;
}

// *****************************************************************************
struct ReLUBackwardFunc: UnaryBackwardFunc<ReLUBackwardFunc> {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        shared_ptr<TensorStorage> inp = m_saved_tensors[0];
        shared_ptr<TensorStorage> res = mul(greater_then(inp, Tensor::get_const(0.).storage()), out_grad);
        m_input_nodes[0]->acc_grad(res);
    }
};

struct ExpBackwardFunc: UnaryBackwardFunc<ExpBackwardFunc> {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        shared_ptr<TensorStorage> out = m_saved_tensors[0];
        shared_ptr<TensorStorage> res = mul(out, out_grad);
        m_input_nodes[0]->acc_grad(res);
    }
};

struct LogBackwardFunc: UnaryBackwardFunc<LogBackwardFunc> {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        shared_ptr<TensorStorage> inp = m_saved_tensors[0];
        shared_ptr<TensorStorage> res = mul(reciprocal(inp), out_grad);
        m_input_nodes[0]->acc_grad(res);
    }
};

struct SigmoidBackwardFunc: UnaryBackwardFunc<SigmoidBackwardFunc> {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        shared_ptr<TensorStorage> inp = m_saved_tensors[0];
        shared_ptr<TensorStorage> res = mul(mul(sigmoid(inp), sigmoid(neg(inp))), out_grad);
        m_input_nodes[0]->acc_grad(res);
    }
};

struct NegBackwardFunc: UnaryBackwardFunc<NegBackwardFunc> {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        shared_ptr<TensorStorage> res = neg(out_grad);
        m_input_nodes[0]->acc_grad(res);
    }
};

struct CopyBackwardFunc: UnaryBackwardFunc<CopyBackwardFunc> {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        m_input_nodes[0]->acc_grad(out_grad);
    }
};

struct ReciprocalBackwardFunc: UnaryBackwardFunc<ReciprocalBackwardFunc> {
    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        shared_ptr<TensorStorage> inp = m_saved_tensors[0];
        shared_ptr<TensorStorage> res = neg(reciprocal(mul(inp, inp)));
        m_input_nodes[0]->acc_grad(res);
    }
};

Tensor unary_op(UnaryOpMode mode, Tensor& x) {
    if(mode == UnaryOpMode::COPY)
        return copy_op(x.storage());
    Tensor res = Tensor(unary_op(mode, x.storage()));
    if(x.need_grad()) {
        shared_ptr<GraphNode> out_node = res.graph_node();
        shared_ptr<GraphNode> x_node = x.graph_node();
        shared_ptr<BackwardFunc> func;
        switch(mode) {
        case UnaryOpMode::RELU:
            func = ReLUBackwardFunc::make(x_node);
            func->m_saved_tensors.push_back(x.storage());
            break;
        case UnaryOpMode::EXP:
            func = ExpBackwardFunc::make(x_node);
            func->m_saved_tensors.push_back(res.storage());
            break;
        case UnaryOpMode::LOG:
            func = LogBackwardFunc::make(x_node);
            func->m_saved_tensors.push_back(x.storage());
            break;
        case UnaryOpMode::SIGMOID:
            func = SigmoidBackwardFunc::make(x_node);
            func->m_saved_tensors.push_back(x.storage());
            break;
        case UnaryOpMode::NEG:
            func = NegBackwardFunc::make(x_node);
            break;
        case UnaryOpMode::COPY:
            func = CopyBackwardFunc::make(x_node);
            break;
        case UnaryOpMode::RECIPROCAL:
            func = ReciprocalBackwardFunc::make(x_node);
            func->m_saved_tensors.push_back(x.storage());
            break;
        default:
            break;
        }
        out_node->set_backward_func(func);
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

Tensor equal(Tensor& x, Tensor& y) {
    return binary_op(BinaryOpMode::EQ, x, y);
}

Tensor less_then(Tensor& x, Tensor& y) {
    return binary_op(BinaryOpMode::LT, x, y);
}

Tensor less_equal(Tensor& x, Tensor& y) {
    return binary_op(BinaryOpMode::LE, x, y);
}

Tensor greater_then(Tensor& x, Tensor& y) {
    return binary_op(BinaryOpMode::GT, x, y);
}

Tensor greater_equal(Tensor& x, Tensor& y) {
    return binary_op(BinaryOpMode::GE, x, y);
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

namespace intl {

}

}
