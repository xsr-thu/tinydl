#include <cassert>
#include "tensor.h"
#include "elementwise.h"

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
};


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
        default:
            break;
        }
    }
}

__global__ void kernel_binary_op(float *out, float *in, size_t n, UnaryOpMode mode) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n) {
        switch(mode) {
        case UnaryOpMode::RELU:
            out[idx] = in[idx] > 0.0f ? in[idx]: 0.0f;
            break;
        case UnaryOpMode::LOG:
            out[idx] = logf(in[idx]);
            break;
        case UnaryOpMode::EXP:
            out[idx] = expf(in[idx]);
            break;
        case UnaryOpMode::SIGMOID:
            out[idx] = 1.f / (1 + expf(in[idx]));
            break;
        default:
            break;
        }
    }
}

Tensor binary_op(BinaryOpMode mode, Tensor x, Tensor y) {
    assert(x.size() == y.size());
    assert(x.dim() == y.dim());
    for(size_t i=0; i<x.dim(); i++)
        assert(x.m_storage->m_shape[i] == y.m_storage->m_shape[i]);

    float *res;
    cudaMalloc(&res, sizeof(float) * x.size());

    int block_size = 128;
    int n_block = (x.size() + block_size - 1) / block_size;
    kernel_binary_op<<<n_block, block_size>>>(res, x.m_storage->m_data, y.m_storage->m_data, x.size(), mode);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error\n");
    }
    return Tensor(res, x.m_storage->m_shape, x.m_storage->m_strides);
}


Tensor unary_op(UnaryOpMode mode, Tensor x) {
    float *res;
    cudaMalloc(&res, sizeof(float) * x.size());

    int block_size = 128;
    int n_block = (x.size() + block_size - 1) / block_size;
    kernel_binary_op<<<n_block, block_size>>>(res, x.m_storage->m_data, x.size(), mode);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error\n");
    }
    return Tensor(res, x.m_storage->m_shape, x.m_storage->m_strides);
}


namespace opr {

Tensor add(Tensor x, Tensor y) {
    return binary_op(BinaryOpMode::ADD, x, y);
}

Tensor sub(Tensor x, Tensor y) {
    return binary_op(BinaryOpMode::SUB, x, y);
}

Tensor mul(Tensor x, Tensor y) {
    return binary_op(BinaryOpMode::MUL, x, y);
}

Tensor div(Tensor x, Tensor y) {
    return binary_op(BinaryOpMode::DIV, x, y);
}


Tensor relu(Tensor x) {
    return unary_op(UnaryOpMode::RELU, x);
}

Tensor log(Tensor x) {
    return unary_op(UnaryOpMode::LOG, x);
}

Tensor exp(Tensor x) {
    return unary_op(UnaryOpMode::EXP, x);
}

Tensor sigmoid(Tensor x) {
    return unary_op(UnaryOpMode::SIGMOID, x);
}

}
