#ifndef __OPR_UTILS_H_
#define __OPR_UTILS_H_

#include <sstream>
#include <cuda_runtime.h>
#include "tensor.h"



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


struct TensorFormat{
    static const size_t MAX_DIM=7;

    size_t size;
    size_t dim;
    size_t shape[MAX_DIM];
    size_t strides[MAX_DIM];

    static TensorFormat* make_cuda_tensor_format(Tensor tensor) {
        TensorFormat format{tensor.size(), tensor.dim()};
        for(size_t i=0; i<tensor.m_shape.size(); i++) {
            format.shape[i] = tensor.m_shape[i];
            format.strides[i] = tensor.m_strides[i];
        }
        TensorFormat *dev_format;
        cudaMalloc(&dev_format, sizeof(TensorFormat));
        cudaMemcpy(dev_format, &format, sizeof(TensorFormat), cudaMemcpyHostToDevice);
        return dev_format;
    }

    static TensorFormat* make_cuda_tensor_format(vector<size_t> &shape, vector<size_t> &strides) {
        size_t size = 1;
        for(size_t s: shape) {
            size *= s;
        }
        TensorFormat format{size, shape.size()};
        for(size_t i=0; i<shape.size(); i++) {
            format.shape[i] = shape[i];
            format.strides[i] = strides[i];
        }
        TensorFormat *dev_format;
        cudaMalloc(&dev_format, sizeof(TensorFormat));
        cudaMemcpy(dev_format, &format, sizeof(TensorFormat), cudaMemcpyHostToDevice);
        return dev_format;
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

    void release() {
        cudaFree(this);
    }
};




#endif
