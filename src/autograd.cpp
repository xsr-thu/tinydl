#include "autograd.h"
#include "opr/elementwise.h"


void GraphNode::acc_grad(shared_ptr<TensorStorage> grad) {
    // FIXME: broadcast
    if(!m_require_grad && !m_need_grad)
        return;
    if(!m_grad_storage) {
        bool layout_is_same = true;
        if(m_shape.size() == grad->m_shape.size()) {
            for(int i=0; i < m_shape.size(); i++) {
                // TODO: fix non-contiougous layout
                // add Layout abstract
                if(m_shape[i] != grad->m_shape[i]) {
                    layout_is_same = false;
                    break;
                }
            }
        } else {
            layout_is_same = false;
        }
        if (layout_is_same) {
            m_grad_storage = opr::intl::copy(grad);
        } else {
            m_grad_storage = TensorStorage::zeros(m_shape);
            m_grad_storage = opr::intl::add(m_grad_storage, grad);
        }
    } else {
        m_grad_storage = opr::intl::add(m_grad_storage, grad);
    }
}


void GraphNode::release() {
    if(!m_require_grad)
        m_grad_storage.reset();
}



using NodePtr = shared_ptr<GraphNode>;


void graph_dfs(unordered_set<NodePtr> &visited, unordered_map<NodePtr, unordered_set<NodePtr>> &in2out, vector<NodePtr> &leafs, NodePtr node) {
    if(visited.count(node))
        return;
    // printf("graph_dfs %zu\n", node->m_id);
    visited.insert(node);

    if(!node->m_require_grad && !node->m_need_grad)
        return;
    if(!node->m_backward_func || !node->m_backward_func->m_input_nodes.size()) {
        leafs.push_back(node);
        return;
    }
    for(auto& n: node->m_backward_func->m_input_nodes) {
        in2out[n].insert(node);
        graph_dfs(visited, in2out, leafs, n);
    }
}


void topo_sort(unordered_set<NodePtr> &visited, unordered_map<NodePtr, unordered_set<NodePtr>> &in2out, vector<NodePtr> &topo_seq, NodePtr node, bool is_leaf) {
    if(visited.count(node))
        return;
    visited.insert(node);

    if(in2out.count(node)) {
        for(auto& n: in2out[node]) {
            topo_sort(visited, in2out, topo_seq, n, false);
        }
    }
    // printf("topo sort id=%zu\n", node->m_id);
    if(!is_leaf)
        topo_seq.push_back(node);
}


void backprop(shared_ptr<GraphNode> output_node) {
    unordered_map<NodePtr, unordered_set<NodePtr>> in2out;
    vector<NodePtr> leafs;
    vector<NodePtr> topo_seq;
    unordered_set<NodePtr> visited;

    graph_dfs(visited, in2out, leafs, output_node);
    visited.clear();

    // printf("graph_dfs, leafs: %zu, in2out: %zu\n", leafs.size(), in2out.size());
    // for(auto& p: in2out) {
    //     printf("node: %zu -->", p.first->m_id);
    //     for(auto& n: p.second) {
    //         printf("%zu, ", n->m_id);
    //     }
    //     printf("\n");
    // }
    // printf("leafs: ");
    // for(auto&n: leafs) {
    //     printf("%zu ", n->m_id);
    // }
    // printf("\n");

    for(auto& n: leafs) {
        topo_sort(visited, in2out, topo_seq, n, true);
    }

    // printf("topo seq: %zu\n", topo_seq.size());
    // for(auto&n : topo_seq) {
    //     printf("%zu ", n->m_id);
    // }
    // printf("\n");

    for(auto& n: topo_seq) {
        // printf("-- backward -- %zu\n", n->m_id);
        n->m_backward_func->operator()(n);
    }
}
