#include "tensor.h"
#include "opr/elementwise.h"
#include "opr/reduction.h"

size_t Tensor::sm_id = 0;
unordered_map<float, Tensor> Tensor::sm_const_cache;

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
    m_id = sm_id++;
    // printf("Tensor ctor: %zu\n", m_id);
}

Tensor::Tensor(float *data, vector<size_t> shape, vector<size_t> strides) {
    size_t size = 1;
    for(size_t i=0; i<shape.size(); i++)
        size *= shape[i];
    m_storage = make_shared<TensorStorage>(data, size, shape, strides);
    m_id = sm_id++;
    // printf("Tensor ctor: %zu\n", m_id);
}


Tensor& Tensor::get_const(float val) {
    if(!sm_const_cache.count(val)) {
        float *data;
        cudaError_t err;
        err = cudaMalloc(&data, sizeof(float));
        float host[1] = {val};
        err = cudaMemcpy(data, host, sizeof(float), cudaMemcpyHostToDevice);
        vector<size_t> shape;
        vector<size_t> strides;
        shape.push_back(1);
        strides.push_back(sizeof(float));
        Tensor t{data, shape, strides};
        sm_const_cache[val] = t;
    }
    return sm_const_cache[val];
}


Tensor::~Tensor() {
    // cudaFree(m_data);
    // printf("Tensor dtor: %zu\n", m_id);
}

size_t Tensor::size() {
    return m_storage->m_size;
}

size_t Tensor::dim() {
    return m_storage->dim();
}


py::array_t<float> Tensor::to_numpy() {
    auto result = py::array_t<float>(m_storage->m_shape, m_storage->m_strides);
    py::buffer_info buf = result.request();

    float *ptr = static_cast<float*>(buf.ptr);
    cudaMemcpy(ptr, m_storage->data(), sizeof(float) * size(), cudaMemcpyDeviceToHost);
    return result;
}

void Tensor::backward(Tensor &grad) {
    if(!m_graph_node) {
        printf("can not backward!");
    } else {
        m_graph_node->m_grad_storage = grad.m_storage;
        backprop(m_graph_node);
    }
}

shared_ptr<BackwardFunc> Tensor::grad_fn() {
    if(!m_graph_node)
        return nullptr;
    return m_graph_node->m_backward_func;
}

shared_ptr<Tensor> Tensor::grad() {
    if(!m_graph_node) {
        printf("Tensor %zu no grad\n", this->m_id);
        return nullptr;
    }
    return std::make_shared<Tensor>(m_graph_node->m_grad_storage);
}
