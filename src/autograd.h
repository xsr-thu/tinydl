#ifndef __AUTOGRAD_H_
#define __AUTOGRAD_H_

#include <vector>
#include <memory>
#include "tensor.h"
using namespace std;

struct BackwardFunc;
struct TensorStorage;

struct GraphNode {
private:
    bool m_requires_grad;
    vector<size_t> m_shape;
    size_t m_id;
    shared_ptr<TensorStorage> m_grad_storage;
    shared_ptr<BackwardFunc> m_backward_func;

public:
    GraphNode(bool requires_grad, size_t id, const vector<size_t> &shape)
        : m_requires_grad(requires_grad), m_id(id), m_shape(shape) {
    }

    void acc_grad(shared_ptr<TensorStorage> grad);

    inline void zero_grad() {
        m_grad_storage.reset();
    }

    inline const vector<size_t>& shape() {
        return m_shape;
    }

    inline size_t dim() {
        return m_shape.size();
    }

    size_t inputs_number();

    inline void set_grad_storage(shared_ptr<TensorStorage> storage) {
        m_grad_storage = storage;
    }

    void release();

    inline shared_ptr<TensorStorage> grad_storage() {
        return m_grad_storage;
    }

    inline void set_backward_func(shared_ptr<BackwardFunc> func) {
        // printf("backward func setted\n");
        m_backward_func = func;
    }

    shared_ptr<BackwardFunc> backward_func() {
        return m_backward_func;
    }

    inline bool need_grad() {
        return bool(m_backward_func) || m_requires_grad;
    }

    inline bool requires_grad() {
        return m_requires_grad;
    }

    inline void set_requires_grad(bool required) {
        m_requires_grad = required;
    }

    ~GraphNode() {
        // printf("GraphNode dtor\n");
    }
};


struct BackwardFunc {
    std::vector<std::shared_ptr<GraphNode>> m_input_nodes;
    std::vector<std::shared_ptr<TensorStorage>> m_saved_tensors;

    void operator()(std::shared_ptr<GraphNode> output_node) {
        backward_func(output_node);
        output_node->release();
        m_saved_tensors.clear();
    }

    virtual void backward_func(std::shared_ptr<GraphNode> output_node) = 0;
};



void backprop(shared_ptr<GraphNode> output_node);


#endif
