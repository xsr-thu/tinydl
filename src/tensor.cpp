#include "tensor.h"
#include "opr/elementwise.h"
#include "opr/reduction.h"


Tensor::Tensor(py::array_t<float> arr) {
    py::buffer_info info = arr.request();
    size_t size = 1;
    vector<size_t> shape;
    vector<size_t> strides;

    for(size_t i=0; i<info.shape.size(); i++) {
        shape.push_back(info.shape[i]);
        size *= info.shape[i];
    }
    for(size_t i=0; i<info.strides.size(); i++) {
        strides.push_back(info.strides[i]);
    }
    float *ptr;
    cudaMalloc(&ptr, sizeof(float)*size);
    // memcpy(m_data, (float*)info.ptr, sizeof(float) * size);
    cudaMemcpy(ptr, (float*)info.ptr, sizeof(float)*size, cudaMemcpyHostToDevice);
    m_storage = make_shared<TensorStorage>(ptr, size, shape, strides);
}

Tensor::Tensor(float *data, vector<size_t> shape, vector<size_t> strides) {
    size_t size = 1;
    for(size_t i=0; i<shape.size(); i++)
        size *= shape[i];
    m_storage = make_shared<TensorStorage>(data, size, shape, strides);
}

Tensor::~Tensor() {
    // cudaFree(m_data);
}

size_t Tensor::size() {
    return m_storage->m_size;
}

size_t Tensor::dim() {
    return m_storage->m_size;
}


py::array_t<float> Tensor::to_numpy() {
    auto result = py::array_t<float>(m_storage->m_shape, m_storage->m_strides);
    py::buffer_info buf = result.request();

    float *ptr = static_cast<float*>(buf.ptr);
    cudaMemcpy(ptr, m_storage->m_data, sizeof(float) * size(), cudaMemcpyDeviceToHost);
    return result;
}
