#include "trans_layout.h"
#include "opr_utils.h"
#include "tensor.h"
#include <cassert>


struct ViewBackwardFunc: BackwardFunc {
    vector<size_t> m_old_shape;
    vector<size_t> m_old_strides;

    static std::shared_ptr<BackwardFunc> make(shared_ptr<GraphNode> x, std::vector<size_t> &shape, std::vector<size_t> &strides){
        shared_ptr<ViewBackwardFunc> func = make_shared<ViewBackwardFunc>();
        func->m_input_nodes.push_back(x);
        func->m_old_shape = shape;
        func->m_old_strides = strides;
        return func;
    }

    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->m_grad_storage;
        shared_ptr<TensorStorage> new_storage = make_shared<TensorStorage>(
                out_grad, m_old_shape, m_old_strides);
        m_input_nodes[0]->acc_grad(new_storage);
    }
};

namespace opr {

Tensor view(Tensor& x, std::vector<size_t> &new_shape) {
    std::shared_ptr<TensorStorage> old_storage = x.m_storage;
    std::vector<size_t> &old_strides = old_storage->m_strides;
    std::vector<size_t> &old_shape = old_storage->m_shape;

    size_t new_size = 1;
    std::vector<size_t> new_strides(new_shape.size());
    for(int i=new_shape.size() - 1; i >= 0; --i) {
        new_strides[i] = new_size * sizeof(float);
        new_size *= new_shape[i];
    }

    assert(new_size == old_storage->size());

    for(int i=1; i<old_strides.size(); i++) {
        assert(old_strides[i-1] >= old_strides[i]);
    }

    std::shared_ptr<TensorStorage> new_storage = std::make_shared<TensorStorage>(
            old_storage, new_shape, new_strides);

    Tensor new_tensor = Tensor(new_storage);

    if(x.m_require_grad || x.m_need_grad) {
        printf("set backward fun for view\n");
        shared_ptr<GraphNode> out_node = new_tensor.graph_node();
        shared_ptr<GraphNode> x_node = x.graph_node();
        shared_ptr<BackwardFunc> func = ViewBackwardFunc::make(x_node, old_shape, old_strides);

        out_node->set_backward_func(func);
        out_node->m_need_grad = true;
        new_tensor.m_need_grad = true;
    }
    return new_tensor;
}

}
