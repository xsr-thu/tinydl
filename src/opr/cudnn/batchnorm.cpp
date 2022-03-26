#include "batchnorm.h"
#include "common.h"

shared_ptr<TensorStorage> batchnorm_forward(
        shared_ptr<TensorStorage> data,
        shared_ptr<TensorStorage> scale,
        shared_ptr<TensorStorage> bias,
        shared_ptr<TensorStorage> running_mean,
        shared_ptr<TensorStorage> running_var,
        bool is_train) {
    const vector<size_t> &data_shape = data->shape();
    vector<size_t> other_shape(4);
    other_shape[0] = other_shape[2] = other_shape[3] = 1;
    other_shape[1] = data_shape[1];

    // TODO: Check stride

    cudnnHandle_t* handle = HandleManager::get()->cudnn_handle();

    cudnnTensorDescriptor_t xDesc, bnOtherDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&bnOtherDesc);

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc,
                CUDNN_TENSOR_NCHW,
                CUDNN_DATA_FLOAT,
                data_shape[0],
                data_shape[1],
                data_shape[2],
                data_shape[3]
                ));

    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bnOtherDesc, xDesc, CUDNN_BATCHNORM_SPATIAL));

    float *dev_output;
    cudaMalloc(&dev_output, sizeof(float) * data->size());

    float alpha=1., beta=0.;
    if (is_train) {
         CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
                    *handle,
                    CUDNN_BATCHNORM_SPATIAL,
                    &alpha, &beta,
                    xDesc, data->data(),
                    xDesc, dev_output,
                    bnOtherDesc,
                    scale->data(), bias->data(),
                    0.001,
                    running_mean->data(), running_var->data(),
                    CUDNN_BN_MIN_EPSILON,
                    nullptr, nullptr
                    ));
    } else {
        CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
                    *handle,
                    CUDNN_BATCHNORM_SPATIAL,
                    &alpha, &beta,
                    xDesc, data->data(),
                    xDesc, dev_output,
                    bnOtherDesc,
                    scale->data(), bias->data(),
                    running_mean->data(),
                    running_var->data(),
                    CUDNN_BN_MIN_EPSILON));
    }

    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(bnOtherDesc);
    return make_shared<TensorStorage>(dev_output, data->size(), data->shape(), data->strides());
}


vector<shared_ptr<TensorStorage>> batchnorm_backward(
        shared_ptr<TensorStorage> data,
        shared_ptr<TensorStorage> grad,
        shared_ptr<TensorStorage> scale,
        shared_ptr<TensorStorage> bias) {

    const vector<size_t> &data_shape = data->shape();
    vector<size_t> other_shape(4);
    other_shape[0] = other_shape[2] = other_shape[3] = 1;
    other_shape[1] = data_shape[1];

    // TODO: Check stride

    cudnnHandle_t* handle = HandleManager::get()->cudnn_handle();

    cudnnTensorDescriptor_t xDesc, bnOtherDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&bnOtherDesc);

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc,
                CUDNN_TENSOR_NCHW,
                CUDNN_DATA_FLOAT,
                data_shape[0],
                data_shape[1],
                data_shape[2],
                data_shape[3]
                ));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bnOtherDesc, xDesc, CUDNN_BATCHNORM_SPATIAL));

    float *dev_grad, *dev_d_bn_scale, *dev_d_bn_bias;
    cudaMalloc(&dev_grad, sizeof(float) * data->size());
    cudaMalloc(&dev_d_bn_scale, sizeof(float) * scale->size());
    cudaMalloc(&dev_d_bn_bias, sizeof(float) * bias->size());

    float alpha=1., beta=0.;
     CUDNN_CHECK(cudnnBatchNormalizationBackward(
                *handle,
                CUDNN_BATCHNORM_SPATIAL,
                &alpha, &beta,
                &alpha, &beta,
                xDesc, data->data(),
                xDesc, grad->data(),
                xDesc, dev_grad,
                bnOtherDesc,
                scale->data(),
                dev_d_bn_scale, dev_d_bn_bias,
                CUDNN_BN_MIN_EPSILON,
                nullptr, nullptr
                ));

    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(bnOtherDesc);

    shared_ptr<TensorStorage> dx = make_shared<TensorStorage>(dev_grad, data->size(), data->shape(), data->strides());
    shared_ptr<TensorStorage> d_bn_scale = make_shared<TensorStorage>(dev_d_bn_scale, scale->size(), scale->shape(), scale->strides());
    shared_ptr<TensorStorage> d_bn_bias = make_shared<TensorStorage>(dev_d_bn_bias, scale->size(), scale->shape(), scale->strides());
    return vector<shared_ptr<TensorStorage>>{dx, d_bn_scale, d_bn_bias};
}


struct BatchNormOpBackwarFunc: BackwardFunc {

    static std::shared_ptr<BackwardFunc> make(shared_ptr<GraphNode> x, shared_ptr<GraphNode> scale, shared_ptr<GraphNode> bias) {
        shared_ptr<BatchNormOpBackwarFunc> func = make_shared<BatchNormOpBackwarFunc>();
        func->m_input_nodes.push_back(x);
        func->m_input_nodes.push_back(scale);
        func->m_input_nodes.push_back(bias);
        return func;
    }

    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->grad_storage();
        shared_ptr<TensorStorage> data = m_saved_tensors[0];
        shared_ptr<TensorStorage> scale = m_saved_tensors[1];
        shared_ptr<TensorStorage> bias = m_saved_tensors[2];

        vector<shared_ptr<TensorStorage>> grads = batchnorm_backward(data, out_grad, scale, bias);

        m_input_nodes[0]->acc_grad(grads[0]);
        m_input_nodes[1]->acc_grad(grads[1]);
        m_input_nodes[2]->acc_grad(grads[2]);
    }
};




namespace opr {

Tensor batchnorm(Tensor& data, Tensor &scale, Tensor &bias, Tensor &running_mean, Tensor &running_var, bool is_train) {
    shared_ptr<TensorStorage> s = batchnorm_forward(
            data.storage(), scale.storage(), bias.storage(), running_mean.storage(), running_var.storage(), is_train);

    Tensor res(s);
    if(!is_train)
        return res;

    if(data.need_grad() || scale.need_grad() || bias.need_grad()) {
        shared_ptr<GraphNode> x_node = data.graph_node();
        shared_ptr<GraphNode> scale_node = scale.graph_node();
        shared_ptr<GraphNode> bias_node = bias.graph_node();
        shared_ptr<GraphNode> out_node = res.graph_node();
        shared_ptr<BackwardFunc> func = BatchNormOpBackwarFunc::make(x_node, scale_node, bias_node);

        func->m_saved_tensors.push_back(data.storage());
        func->m_saved_tensors.push_back(scale.storage());
        func->m_saved_tensors.push_back(bias.storage());

        out_node->set_backward_func(func);
    }
    return res;
}

}
