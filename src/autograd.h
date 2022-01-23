#ifndef __AUTOGRAD_H_
#define __AUTOGRAD_H_

#include <vector>
#include <memory>
#include "tensor.h"
using namespace std;

struct BackwardFunc;
struct TensorStorage;

struct GraphNode {
    bool m_require_grad;
    bool m_need_grad;
    size_t m_id;
    shared_ptr<TensorStorage> m_grad_storage;
    shared_ptr<BackwardFunc> m_backward_func;

    GraphNode(bool require_grad, bool need_grad, size_t id)
        : m_require_grad(require_grad), m_need_grad(need_grad), m_id(id) {
        // printf("GraphNode ctor\n");
    }

    void acc_grad(shared_ptr<TensorStorage> grad);

    void release();

    void set_backward_func(shared_ptr<BackwardFunc> func) {
        // printf("backward func setted\n");
        m_backward_func = func;
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
