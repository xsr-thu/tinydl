#ifndef __OPR_UTILS_H_
#define __OPR_UTILS_H_

#include <sstream>
#include <cuda_runtime.h>
#include "tensor.h"
#include <memory>


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


template<typename T>
struct ToStringTrait {
    static std::string to_string(T &data);
};

template<typename T>
struct ToStringTrait<std::vector<T>> {
    static std::string to_string(std::vector<T> &data) {
        std::stringstream ss;
        ss << "(";
        for(int i=0;i<data.size(); i++) {
            ss << data[i];
            if(i!=data.size() - 1)
                ss << ", ";
        }
        ss << ")";
        return ss.str();
    }
};


template<typename T>
std::string to_string(T &data) {
    return ToStringTrait<T>::to_string(data);
}

struct TensorFormat;

using CudaTensorFormatPtr = std::shared_ptr<TensorFormat>;

struct TensorFormat{
    static const size_t MAX_DIM=7;

    size_t size;
    size_t dim;
    size_t shape[MAX_DIM];
    size_t strides[MAX_DIM];

    static CudaTensorFormatPtr make_cuda_tensor_format(Tensor tensor) {
        TensorFormat format{tensor.size(), tensor.dim()};
        for(size_t i=0; i<tensor.dim(); i++) {
            format.shape[i] = tensor.shape()[i];
            format.strides[i] = tensor.strides()[i];
        }
        TensorFormat *dev_format;
        cudaMalloc(&dev_format, sizeof(TensorFormat));
        cudaMemcpy(dev_format, &format, sizeof(TensorFormat), cudaMemcpyHostToDevice);
        return std::shared_ptr<TensorFormat>(dev_format, [](TensorFormat* ptr){cudaFree(ptr);});
    }

    static CudaTensorFormatPtr make_cuda_tensor_format(const vector<size_t> &shape, const vector<size_t> &strides) {
        size_t size = 1;
        cudaError_t err;
        for(size_t s: shape) {
            size *= s;
        }
        TensorFormat format{size, shape.size()};
        for(size_t i=0; i<shape.size(); i++) {
            format.shape[i] = shape[i];
            format.strides[i] = strides[i];
        }
        TensorFormat *dev_format;
        err = cudaMalloc(&dev_format, sizeof(TensorFormat));
        if (err != cudaSuccess) {
            printf("cuda error %s\n", cudaGetErrorString(err));
            throw std::runtime_error(cudaGetErrorString(err));
        }
        err = cudaMemcpy(dev_format, &format, sizeof(TensorFormat), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("cuda error %s\n", cudaGetErrorString(err));
            throw std::runtime_error(cudaGetErrorString(err));
        }
        // printf("make format: %zu %p\n", format.shape[0], dev_format);
        return std::shared_ptr<TensorFormat>(dev_format, [](TensorFormat* ptr){cudaFree(ptr);});
    }

    std::string to_string() {
        std::stringstream ss;
        printf("tostring dim=%lu\n", dim);
        ss << "<Tensor shape=(";
        for(int i=0; i < dim; i++) {
            if(i>0)
                ss << ", ";
            ss << shape[i];
        }
        ss << "), strides=(";
        for(int i=0; i < dim; i++) {
            if(i>0)
                ss << ", ";
            ss << strides[i];
        }
        ss << ")>";
        return ss.str();
    }
};




#endif
