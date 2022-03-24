#include <string>
#include <iostream>
#include <cudnn.h>
#include <cassert>
#include "conv.h"
#include "tensor.h"
#include "../opr_utils.h"
#include "../../autograd.h"


void CUDNN_CHECK(cudnnStatus_t s) {
    if(s != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "CUDNN: %s\n", cudnnGetErrorString(s));
        assert(0);
    }
}


shared_ptr<TensorStorage> conv2d_forward(shared_ptr<TensorStorage> data, shared_ptr<TensorStorage> weight, size_t padding, size_t stride) {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    vector<size_t>& data_shape = data->m_shape;
    vector<size_t>& weight_shape = weight->m_shape;
    vector<size_t> output_shape(4);
    output_shape[0] = data_shape[0];
    output_shape[1] = weight_shape[0];
    size_t kh = weight_shape[2];
    size_t kw = weight_shape[3];
    size_t pad_h = 1, pad_w = 1;
    size_t stride_h = stride, stride_w = stride;
    output_shape[2] = (data_shape[2] + pad_h + pad_h - kh + 1) / stride_h;
    output_shape[3] = (data_shape[2] + pad_w + pad_w - kw + 1) / stride_w;

    // fprintf(stderr, "padding=%zu stride=%zu\n", padding, stride);
    // fprintf(stderr, "inputs: %s\n", to_string(data_shape).c_str());
    // fprintf(stderr, "weight: %s\n", to_string(weight_shape).c_str());
    // fprintf(stderr, "outputs: %s\n", to_string(output_shape).c_str());

    cudnnTensorDescriptor_t input_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, 
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            data_shape[0],
            data_shape[1],
            data_shape[2],
            data_shape[3]));

    cudnnTensorDescriptor_t output_desc;
    cudnnCreateTensorDescriptor(&output_desc);
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            output_shape[0],
            output_shape[1],
            output_shape[2],
            output_shape[3]));

    cudnnFilterDescriptor_t kernel_desc;
    cudnnCreateFilterDescriptor(&kernel_desc);
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(kernel_desc, 
                CUDNN_DATA_FLOAT, 
                CUDNN_TENSOR_NCHW,
                weight_shape[0],
                weight_shape[1],
                weight_shape[2],
                weight_shape[3]));

    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
            padding, // pad_h 
            padding, // pad_w
            stride, // stride u
            stride, // stride v
            1, // dilation_h
            1, // dilation_w
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    size_t space_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, kernel_desc, conv_desc, output_desc,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &space_size));

    void* workspace = nullptr;
    cudaMalloc(&workspace, space_size);

    float *dev_output;
    cudaMalloc(&dev_output, output_shape[0]*output_shape[1]*output_shape[2]*output_shape[3]*sizeof(float));

    float alpha = 1.0;
    float beta = .0;

    CUDNN_CHECK(cudnnConvolutionForward(handle, &alpha, 
            input_desc, data->data(),
            kernel_desc, weight->data(),
            conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, workspace, space_size,
            &beta, output_desc, dev_output));
    
    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(kernel_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(handle);

    vector<size_t> out_strides(4);
    size_t size = sizeof(float);
    for(int i=3;i>=0; i--) {
        out_strides[i] = size;
        size *= output_shape[i];
    }
    return make_shared<TensorStorage>(dev_output, size / sizeof(float), output_shape, out_strides);
}


shared_ptr<TensorStorage> conv2d_bwd_data(vector<size_t> &data_shape,shared_ptr<TensorStorage> grad, shared_ptr<TensorStorage> weight,
        size_t padding, size_t stride) {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    vector<size_t>& weight_shape = weight->m_shape;
    vector<size_t> grad_shape = grad->m_shape;

    cudnnTensorDescriptor_t grad_desc;
    cudnnCreateTensorDescriptor(&grad_desc);
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(grad_desc, 
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            grad_shape[0],
            grad_shape[1],
            grad_shape[2],
            grad_shape[3]));

    cudnnTensorDescriptor_t data_desc;
    cudnnCreateTensorDescriptor(&data_desc);
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(data_desc,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            data_shape[0],
            data_shape[1],
            data_shape[2],
            data_shape[3]));

    cudnnFilterDescriptor_t kernel_desc;
    cudnnCreateFilterDescriptor(&kernel_desc);
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
            weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]));

    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
            padding, // pad_h 
            padding, // pad_w
            stride, // stride u
            stride, // stride v
            1, // dilation_h
            1, // dilation_w
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    size_t space_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, kernel_desc, grad_desc, conv_desc, data_desc,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, &space_size));

    void* workspace = nullptr;
    cudaMalloc(&workspace, space_size);

    float *dev_data_grad;
    cudaMalloc(&dev_data_grad, data_shape[0]*data_shape[1]*data_shape[2]*data_shape[3]*sizeof(float));

    float alpha = 1.0;
    float beta = 0.0;

    CUDNN_CHECK(cudnnConvolutionBackwardData(handle, &alpha, 
            kernel_desc, weight->data(),
            grad_desc, grad->data(),
            conv_desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, workspace, space_size,
            &beta, data_desc, dev_data_grad));
    
    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(grad_desc);
    cudnnDestroyTensorDescriptor(data_desc);
    cudnnDestroyFilterDescriptor(kernel_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(handle);

    vector<size_t> data_strides(4);
    size_t size = sizeof(float);
    for(int i=3;i>=0; i--) {
        data_strides[i] = size;
        size *= data_shape[i];
    }
    return make_shared<TensorStorage>(dev_data_grad, size / sizeof(float), data_shape, data_strides);
}


shared_ptr<TensorStorage> conv2d_bwd_filter(vector<size_t> &weight_shape, shared_ptr<TensorStorage> grad, shared_ptr<TensorStorage> data,
        size_t padding, size_t stride) {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    vector<size_t>& data_shape = data->m_shape;
    vector<size_t> grad_shape = grad->m_shape;

    cudnnTensorDescriptor_t grad_desc;
    cudnnCreateTensorDescriptor(&grad_desc);
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(grad_desc, 
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            grad_shape[0],
            grad_shape[1],
            grad_shape[2],
            grad_shape[3]));

    cudnnTensorDescriptor_t data_desc;
    cudnnCreateTensorDescriptor(&data_desc);
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(data_desc,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            data_shape[0],
            data_shape[1],
            data_shape[2],
            data_shape[3]));

    cudnnFilterDescriptor_t kernel_desc;
    cudnnCreateFilterDescriptor(&kernel_desc);
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
            weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]));

    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
            padding, // pad_h 
            padding, // pad_w
            stride, // stride u
            stride, // stride v
            1, // dilation_h
            1, // dilation_w
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    size_t space_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, data_desc, grad_desc, conv_desc, kernel_desc,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, &space_size));

    void* workspace = nullptr;
    cudaMalloc(&workspace, space_size);

    float *dev_kernel_grad;
    cudaMalloc(&dev_kernel_grad, weight_shape[0]*weight_shape[1]*weight_shape[2]*weight_shape[3]*sizeof(float));

    float alpha = 1.0;
    float beta = 0.0;

    CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle, &alpha, 
            data_desc, data->data(),
            grad_desc, grad->data(),
            conv_desc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, workspace, space_size,
            &beta, kernel_desc, dev_kernel_grad));
    
    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(grad_desc);
    cudnnDestroyTensorDescriptor(data_desc);
    cudnnDestroyFilterDescriptor(kernel_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(handle);

    vector<size_t> weight_grad_strides(4);
    size_t size = sizeof(float);
    for(int i=3;i>=0; i--) {
        weight_grad_strides[i] = size;
        size *= weight_shape[i];
    }
    return make_shared<TensorStorage>(dev_kernel_grad, size / sizeof(float), weight_shape, weight_grad_strides);
}


struct Conv2DOpBackwarFunc: BackwardFunc {
    size_t m_padding;
    size_t m_stride;

    static std::shared_ptr<BackwardFunc> make(shared_ptr<GraphNode> x, shared_ptr<GraphNode> y, size_t padding, size_t stride) {
        shared_ptr<Conv2DOpBackwarFunc> func = make_shared<Conv2DOpBackwarFunc>();
        func->m_padding = padding;
        func->m_stride = stride;
        func->m_input_nodes.push_back(x);
        func->m_input_nodes.push_back(y);
        return func;
    }

    void backward_func(shared_ptr<GraphNode> out_node) override {
        shared_ptr<TensorStorage> out_grad = out_node->m_grad_storage;
        shared_ptr<TensorStorage> data = m_saved_tensors[0];
        shared_ptr<TensorStorage> weight = m_saved_tensors[1];
        
        shared_ptr<TensorStorage> g1 = conv2d_bwd_data(
                this->m_input_nodes[0]->m_shape,
                out_grad,
                weight,
                m_padding,
                m_stride);
        shared_ptr<TensorStorage> g2 = conv2d_bwd_filter(
                this->m_input_nodes[1]->m_shape,
                out_grad,
                data,
                m_padding,
                m_stride);
        
        m_input_nodes[0]->acc_grad(g1);
        m_input_nodes[1]->acc_grad(g2);
    }
};


namespace opr {

Tensor conv2d(Tensor& data, Tensor& weight, size_t padding, size_t stride) {
    Tensor res(conv2d_forward(data.m_storage, weight.m_storage, padding, stride));
    
    if(data.m_need_grad || weight.m_need_grad || data.m_require_grad || weight.m_require_grad) {
        shared_ptr<GraphNode> x_node = data.graph_node();
        shared_ptr<GraphNode> y_node = weight.graph_node();
        shared_ptr<GraphNode> out_node = res.graph_node();
        shared_ptr<BackwardFunc> func = Conv2DOpBackwarFunc::make(x_node, y_node, padding, stride);
    
        func->m_saved_tensors.push_back(data.m_storage);
        func->m_saved_tensors.push_back(weight.m_storage);
        
        out_node->set_backward_func(func);
        out_node->m_need_grad = true;
        res.m_need_grad = true;
    }
    return res;
}

}
