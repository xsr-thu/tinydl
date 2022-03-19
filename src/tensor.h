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

    Tensor(shared_ptr<TensorStorage> storage): m_storage(storage) {
        m_id = sm_id++;
        // printf("Tensor ctor: %zu\n", m_id);
    }

    static Tensor& get_const(float val);

    ~Tensor();

    size_t size();

    size_t dim();

    py::array_t<float> to_numpy();

    void require_grad(bool required) {
        m_require_grad = required;
    }

    void backward(Tensor &grad);

    // void set_graph_node(std::shared_ptr<GraphNode> node) {
    //     m_graph_node = node;
    // }

    std::shared_ptr<GraphNode> graph_node() {
        if(!m_graph_node) {
            // printf("create graph node for %zu(%p)\n", this->m_id, this);
            m_graph_node = make_shared<GraphNode>(m_require_grad, m_need_grad, m_id);
        }
        return m_graph_node;
    }

    shared_ptr<BackwardFunc> grad_fn();

    shared_ptr<Tensor> grad();

};




#endif
