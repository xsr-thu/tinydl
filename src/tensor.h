#ifndef __TENSOR_H_
#define __TENSOR_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <memory>
#include "autograd.h"
#include "dtype.h"

namespace py = pybind11;
using namespace std;

struct GraphNode;
struct BackwardFunc;


struct RawTensor {
public:
    RawTensor(void *ptr): m_ptr(ptr) {
    }

    void* ptr() {
        return m_ptr;
    }

    ~RawTensor() {
        if(m_ptr) {
            cudaFree(m_ptr);
        }
    }

private:
    void *m_ptr;
};

template<class T>
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


struct TensorStorage {
public:
    TensorStorage(void* data, size_t size, vector<size_t> shape, vector<size_t> strides, DataType dtype=DataType::Float32)
    : m_dtype(dtype), m_data(std::make_shared<RawTensor>(data)), m_size(size), m_shape(shape), m_strides(strides) {
        // fprintf(stderr, "strides: %s\n", to_string(m_strides).c_str());
    }

    TensorStorage(shared_ptr<TensorStorage> old_storage, vector<size_t> &shape, vector<size_t> &strides, DataType dtype=DataType::Float32)
    : m_dtype(dtype), m_data(old_storage->m_data), m_size(old_storage->size()), m_shape(shape), m_strides(strides) {
        // fprintf(stderr, "strides: %s\n", to_string(m_strides).c_str());
    }

    static shared_ptr<TensorStorage> zeros(vector<size_t> &shape);

    inline size_t dim() {
        return m_shape.size();
    }
    
    inline size_t size() {
        return m_size;
    }

    inline const vector<size_t>& shape() {
        return m_shape;
    }

    inline const vector<size_t>& strides() {
        return m_strides;
    }

    inline void* data() {
        return m_data.get()->ptr();
    }

    template<typename T>
    inline T* data() {
        return static_cast<T*>(m_data.get()->ptr());
    }

    inline DataType dtype() {
        return m_dtype;
    }

    ~TensorStorage() {
        // if(m_data)
        //    cudaFree(m_data);
    }

private:
    DataType m_dtype;
    std::shared_ptr<RawTensor> m_data;
    size_t m_size;
    vector<size_t> m_shape;
    vector<size_t> m_strides;
};


template<>
inline float* TensorStorage::data<float>() {
    assert(m_dtype == DataType::Float32);
    return static_cast<float*>(m_data.get()->ptr());
}


template<>
inline uint64_t* TensorStorage::data<uint64_t>() {
    assert(m_dtype == DataType::UInt64);
    return static_cast<uint64_t*>(m_data.get()->ptr());
}


template<>
inline bool* TensorStorage::data<bool>() {
    assert(m_dtype == DataType::Bool);
    return static_cast<bool*>(m_data.get()->ptr());
}


struct Tensor {
private:
    static size_t sm_id;
    static unordered_map<float, Tensor> sm_const_cache;
    size_t m_id;
    bool m_requires_grad = false;
    shared_ptr<TensorStorage> m_storage;
    shared_ptr<GraphNode> m_graph_node;

public:
    Tensor() {};

    Tensor(py::array_t<float> arr);

    Tensor(float *data, vector<size_t> shape, vector<size_t> strides);

    Tensor(shared_ptr<TensorStorage> storage);

    static Tensor& get_const(float val);
    // static Tensor zeros(vector<size_t> &shape);

    ~Tensor();

    inline size_t size() {
        return m_storage->size();
    }

    inline size_t dim() {
        return m_storage->dim();
    }

    inline size_t id() {
        return m_id;
    }

    inline const vector<size_t>& shape() {
        return m_storage->shape();
    }

    inline const vector<size_t>& strides() {
        return m_storage->strides();
    }

    inline shared_ptr<TensorStorage> storage() {
        return m_storage;
    }

    template<typename T>
    py::array_t<T> to_numpy();

    void set_requires_grad(bool required);

    bool requires_grad() {
        return m_requires_grad;
    }

    bool need_grad();

    void backward(Tensor &grad);

    // void set_graph_node(std::shared_ptr<GraphNode> node) {
    //     m_graph_node = node;
    // }

    std::shared_ptr<GraphNode> graph_node() {
        if(!m_graph_node) {
            // fprintf(stderr, "create graph node for %zu(%p)\n", this->m_id, this);
            // for(int i=0;i< m_storage->m_shape.size(); i++) {
            //     fprintf(stderr, "id=%zu %d %lu\n", m_id, i, m_storage->m_shape[i]);
            // }

            m_graph_node = make_shared<GraphNode>(m_requires_grad, m_id, shape());
        }
        return m_graph_node;
    }

    DataType dtype() {
        return m_storage->dtype();
    }

    shared_ptr<BackwardFunc> grad_fn();

    shared_ptr<Tensor> grad();

    void zero_grad();

    void set_value(Tensor &rhs);
};

#endif
