#include "tensor.h"
#include "opr/elementwise.h"
#include "opr/reduction.h"
#include "opr/opr_utils.h"

size_t Tensor::sm_id = 0;
unordered_map<float, Tensor> Tensor::sm_const_cache;


shared_ptr<TensorStorage> TensorStorage::zeros(vector<size_t> &shape) {
    float *dev;
    size_t n_elem = 1;
    vector<size_t> strides(shape.size());

    for(int i=shape.size() - 1; i>=0; --i) {
        strides[i] = n_elem;
        n_elem *= shape[i];
    }

    cudaMalloc(&dev, sizeof(float) * n_elem);
    cudaMemset(dev, 0., sizeof(float) * n_elem);
    return make_shared<TensorStorage>(dev, n_elem, shape, strides);
}


Tensor::Tensor(py::array_t<float> arr) {
    py::buffer_info info = arr.request();
    size_t n_elem = 1;
    vector<size_t> shape;
    vector<size_t> strides;

    for(size_t i=0; i<info.shape.size(); i++) {
        shape.push_back(info.shape[i]);
        n_elem *= info.shape[i];
    }
    for(size_t i=0; i<info.strides.size(); i++) {
        strides.push_back(info.strides[i] / sizeof(float));
    }
    float *ptr;
    cudaMalloc(&ptr, sizeof(float)*n_elem);
    cudaMemcpy(ptr, (float*)info.ptr, sizeof(float)*n_elem, cudaMemcpyHostToDevice);
    m_storage = make_shared<TensorStorage>(ptr, n_elem, shape, strides, DataType::Float32);
    m_id = sm_id++;
}


Tensor::Tensor(py::array_t<uint64_t> arr) {
    py::buffer_info info = arr.request();
    size_t n_elem = 1;
    vector<size_t> shape;
    vector<size_t> strides;

    for(size_t i=0; i<info.shape.size(); i++) {
        shape.push_back(info.shape[i]);
        n_elem *= info.shape[i];
    }
    for(size_t i=0; i<info.strides.size(); i++) {
        strides.push_back(info.strides[i] / sizeof(uint64_t));
    }
    uint64_t *ptr;
    cudaMalloc(&ptr, sizeof(uint64_t)*n_elem);
    cudaMemcpy(ptr, (uint64_t*)info.ptr, sizeof(uint64_t)*n_elem, cudaMemcpyHostToDevice);
    m_storage = make_shared<TensorStorage>(ptr, n_elem, shape, strides, DataType::UInt64);
    m_id = sm_id++;
}


Tensor::Tensor(py::array_t<int64_t> arr) {
    py::buffer_info info = arr.request();
    size_t n_elem = 1;
    vector<size_t> shape;
    vector<size_t> strides;

    for(size_t i=0; i<info.shape.size(); i++) {
        shape.push_back(info.shape[i]);
        n_elem *= info.shape[i];
    }
    for(size_t i=0; i<info.strides.size(); i++) {
        strides.push_back(info.strides[i] / sizeof(int64_t));
    }
    int64_t *ptr;
    cudaMalloc(&ptr, sizeof(uint64_t)*n_elem);
    cudaMemcpy(ptr, (uint64_t*)info.ptr, sizeof(int64_t)*n_elem, cudaMemcpyHostToDevice);
    m_storage = make_shared<TensorStorage>(ptr, n_elem, shape, strides, DataType::Int64);
    m_id = sm_id++;
}



Tensor::Tensor(py::array_t<bool> arr) {
    py::buffer_info info = arr.request();
    size_t n_elem = 1;
    vector<size_t> shape;
    vector<size_t> strides;

    for(size_t i=0; i<info.shape.size(); i++) {
        shape.push_back(info.shape[i]);
        n_elem *= info.shape[i];
    }
    for(size_t i=0; i<info.strides.size(); i++) {
        strides.push_back(info.strides[i] / sizeof(bool));
    }
    bool *ptr;
    cudaMalloc(&ptr, sizeof(bool)*n_elem);
    cudaMemcpy(ptr, (bool*)info.ptr, sizeof(bool)*n_elem, cudaMemcpyHostToDevice);
    m_storage = make_shared<TensorStorage>(ptr, n_elem, shape, strides, DataType::Bool);
    m_id = sm_id++;
}


Tensor::Tensor(float *data, vector<size_t> shape, vector<size_t> strides) {
    size_t size = 1;
    for(size_t i=0; i<shape.size(); i++)
        size *= shape[i];
    m_storage = make_shared<TensorStorage>(data, size, shape, strides);

    m_id = sm_id++;
    // printf("Tensor ctor: %zu\n", m_id);
    //
    // fprintf(stderr, "Create Tensor(%zu) from pointer shape %s stride %s\n", m_id,  to_string(shape).c_str(), to_string(strides).c_str());
}


Tensor::Tensor(shared_ptr<TensorStorage> storage): m_storage(storage) {
    m_id = sm_id++;
    // printf("Tensor ctor: %zu\n", m_id);
    // fprintf(stderr, "Create Tensor(%zu) with storage shape %s stride %s\n",
    //         m_id,  to_string(m_storage->m_shape).c_str(), to_string(m_storage->m_strides).c_str());
}


void Tensor::set_requires_grad(bool required) {
    m_requires_grad = required;
    if(m_graph_node) {
        m_graph_node->set_requires_grad(required);
    }
}

bool Tensor::need_grad() {
    return m_requires_grad || (m_graph_node && m_graph_node->need_grad());
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
        strides.push_back(1);
        Tensor t{data, shape, strides};
        sm_const_cache[val] = t;
    }
    return sm_const_cache[val];
}


// Tensor zeros(vector<size_t> &shape) {
//     float *dev;
//     size_t size = 1;
//     vector<size_t> strides(shape.size());
//
//     for(int i=shape.size() - 1; i>=0; --i) {
//         strides[i] = size * sizeof(float);
//         size *= shape[i];
//     }
//
//     cudaMalloc(&dev, sizeof(float) * size);
//     cudaMemset(dev, sizeof(float) * size);
//     return Tensor(dev, shape, strides);
// }


Tensor::~Tensor() {
    // cudaFree(m_data);
    // printf("Tensor dtor: %zu\n", m_id);
}


template<typename T>
py::array_t<T> to_numpy_impl(shared_ptr<TensorStorage> &storage) {
    // fprintf(stderr, "To numpy <%zu>: shape %s strides %s size: %lu\n",
    //         m_id,
    //         to_string(m_storage->m_shape).c_str(),
    //         to_string(m_storage->m_strides).c_str(),
    //         size());
    vector<size_t> strides(storage->strides().size());
    for(int i=0; i<strides.size(); i++) {
        strides[i] = storage->strides()[i] * sizeof(T);
    }
    auto result = py::array_t<T>(storage->shape(), strides);
    py::buffer_info buf = result.request();

    T *ptr = static_cast<T*>(buf.ptr);
    cudaMemcpy(ptr, storage->data(), sizeof(T) * storage->size(), cudaMemcpyDeviceToHost);
    return result;
}


template<>
py::array_t<uint64_t> Tensor::to_numpy<uint64_t>() {
    assert(dtype() == DataType::UInt64);
    return to_numpy_impl<uint64_t>(m_storage);
}

template<>
py::array_t<int64_t> Tensor::to_numpy<int64_t>() {
    assert(dtype() == DataType::Int64);
    return to_numpy_impl<int64_t>(m_storage);
}

template<>
py::array_t<float> Tensor::to_numpy<float>() {
    assert(dtype() == DataType::Float32);
    return to_numpy_impl<float>(m_storage);
}

template<>
py::array_t<bool> Tensor::to_numpy<bool>() {
    assert(dtype() == DataType::Bool);
    return to_numpy_impl<bool>(m_storage);
}


void Tensor::backward(Tensor &grad) {
    if(!m_graph_node) {
        fprintf(stderr, "can not backward tensor (%zu)!\n", m_id);
        throw std::runtime_error("Exception: can not backward throw this Tensor");
    } else {
        m_graph_node->set_grad_storage(grad.m_storage);
        backprop(m_graph_node);
    }
}

shared_ptr<BackwardFunc> Tensor::grad_fn() {
    if(!m_graph_node)
        return nullptr;
    return m_graph_node->backward_func();
}

shared_ptr<Tensor> Tensor::grad() {
    if(!m_graph_node) {
        fprintf(stderr, "Tensor %zu does not has grad\n", this->m_id);
        return nullptr;
    }
    return std::make_shared<Tensor>(m_graph_node->grad_storage());
}


void Tensor::zero_grad() {
    if(m_graph_node && m_graph_node->grad_storage()) {
        m_graph_node->zero_grad();
    }
}

void Tensor::set_value(Tensor &rhs) {
    // fprintf(stderr, "%zu %zu %zu %zu\n", m_storage->size(), rhs.m_storage->size(), m_storage->dim(), rhs.m_storage->dim());
    assert(size() == rhs.size());
    assert(dim() == rhs.dim());
    for(int i=0; i < dim(); i++) {
        assert(shape()[i] == rhs.shape()[i]);
    }

    m_storage = opr::intl::copy(rhs.storage());
}
