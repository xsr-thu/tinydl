#ifndef __TENSOR_H_
#define __TENSOR_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <memory>
#include "autograd.h"


namespace py = pybind11;
using namespace std;

struct GraphNode;
struct BackwardFunc;


struct RawTensor {
    float *m_ptr;
    RawTensor(float *ptr): m_ptr(ptr) {
    }

    float* ptr() {
        return m_ptr;
    }

    ~RawTensor() {
        if(m_ptr) {
            cudaFree(m_ptr);
        }
    }
};


struct TensorStorage {
    std::shared_ptr<RawTensor> m_data;
    size_t m_size;
    vector<size_t> m_shape;
    vector<size_t> m_strides;

    TensorStorage(float* data, size_t size, vector<size_t> shape, vector<size_t> strides)
    : m_data(std::make_shared<RawTensor>(data)), m_size(size), m_shape(shape), m_strides(strides) {
    }

    TensorStorage(shared_ptr<TensorStorage> old_storage, vector<size_t> &shape, vector<size_t> &strides)
    : m_data(old_storage->m_data), m_size(old_storage->size()), m_shape(shape), m_strides(strides) {
    }

    static shared_ptr<TensorStorage> zeros(vector<size_t> &shape);

    size_t dim() {
        return m_shape.size();
    }
    
    size_t size() {
        return m_size;
    }

    float* data() {
        return m_data.get()->ptr();
    }

    ~TensorStorage() {
        // if(m_data)
        //    cudaFree(m_data);
    }
};


struct Tensor {
    static size_t sm_id;
    static unordered_map<float, Tensor> sm_const_cache;
    size_t m_id;
    bool m_require_grad = false;
    bool m_need_grad = false;
    shared_ptr<TensorStorage> m_storage;
    shared_ptr<GraphNode> m_graph_node;

    Tensor() {};

    Tensor(py::array_t<float> arr);

    Tensor(float *data, vector<size_t> shape, vector<size_t> strides);

    Tensor(shared_ptr<TensorStorage> storage);

    static Tensor& get_const(float val);
    // static Tensor zeros(vector<size_t> &shape);

    ~Tensor();

    size_t size();

    size_t dim();

    py::array_t<float> to_numpy();

    void require_grad(bool required) {
        m_require_grad = required;
    }

    bool has_grad() {
        return m_require_grad;
    }


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

            m_graph_node = make_shared<GraphNode>(m_require_grad, m_need_grad, m_id, m_storage->m_shape);
        }
        return m_graph_node;
    }

    shared_ptr<BackwardFunc> grad_fn();

    shared_ptr<Tensor> grad();

    void zero_grad();

    void set_value(Tensor &rhs);
};




#endif
