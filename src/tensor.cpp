#include "tensor.h"
#include "elementwise.h"
#include "reduction.h"


Tensor::Tensor(py::array_t<float> arr) {
    py::buffer_info info = arr.request();
    size_t size = 1;

    for(size_t i=0; i<info.shape.size(); i++) {
        m_shape.push_back(info.shape[i]);
        size *= info.shape[i];
    }
    for(size_t i=0; i<info.strides.size(); i++) {
        m_strides.push_back(info.strides[i]);
    }
    m_size = size;
    float *ptr;
    cudaMalloc(&ptr, sizeof(float)*size);
    // memcpy(m_data, (float*)info.ptr, sizeof(float) * size);
    cudaMemcpy(ptr, (float*)info.ptr, sizeof(float)*size, cudaMemcpyHostToDevice);
    m_data.reset(ptr, [](float *p){cudaFree(p);});
}

Tensor::Tensor(float *data, vector<size_t> shape, vector<size_t> strides)
    :m_shape(shape), m_strides(strides) {
    size_t size = 1;
    for(size_t i=0; i<shape.size(); i++)
        size *= shape[i];
    m_size = size;
    m_data.reset(data, [](float *p){cudaFree(p);});
}

Tensor::~Tensor() {
    // cudaFree(m_data);
}

size_t Tensor::size() {
    return m_size;
}

size_t Tensor::dim() {
    return m_shape.size();
}


py::array_t<float> Tensor::to_numpy() {
    auto result = py::array_t<float>(m_shape, m_strides);
    py::buffer_info buf = result.request();

    float *ptr = static_cast<float*>(buf.ptr);
    cudaMemcpy(ptr, m_data.get(), sizeof(float) * m_size, cudaMemcpyDeviceToHost);
    return result;
}
