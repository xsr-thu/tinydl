
#include "opr_utils.h"



__global__ void kernel_matmul(float *out, TensorFormat *out_format,
        float *a, TensorFormat *a_format,
        float *b, TensorFormat *b_format) {
    const size_t BLOCK_DIM = 32;
    size_t inner_x = threadIdx.x;
    size_t inner_y = threadIdx.y;
    size_t batch_idx = blockIdx.z;
    size_t outer_x = blockIdx.x;
    size_t outer_y = blockIdx.y;
    size_t idx_x = outer_x * BLOCK_DIM + inner_x;
    size_t idx_y = outer_y * BLOCK_DIM + inner_y;

    size_t batch_stride_a = a_format->shape[0] == 1 ? 0: a_format->strides[0];
    size_t batch_stride_b = b_format->shape[0] == 1 ? 0: b_format->strides[0];
    size_t batch_stride_o = out_format->shape[0] == 1 ? 0: out_format->strides[0];

    size_t reduction_dim = a_format->shape[2];

    __shared__ float buf_a[32][32], buf_b[32][32];

    float ans = 0.f;
    for(size_t r=0; r < reduction_dim; r+= BLOCK_DIM) {
        size_t idx_a = batch_stride_a * batch_idx + a_format->strides[1] * idx_x + a_format->strides[2] * (r + inner_y);
        size_t idx_b = batch_stride_b * batch_idx + b_format->strides[2] * idx_y + b_format->strides[1] * (r + inner_x);
        
        if(idx_x < a_format->shape[1] && a_format->shape[2])
            buf_a[inner_x][inner_y] = a[idx_a/sizeof(float)];
        else
            buf_a[inner_x][inner_y] = 0.f;
        
        if(idx_x < b_format->shape[1] && b_format->shape[2])
            buf_b[inner_x][inner_y] = b[idx_b/sizeof(float)];
        else
            buf_b[inner_x][inner_y] = 0.f;
        __syncthreads();

        for(int i=0;i<BLOCK_DIM;i++) {
            ans += buf_a[inner_x][i] * buf_b[i][inner_y];
        }
        __syncthreads();
    }
    size_t out_idx = (batch_stride_o * batch_idx + out_format->strides[1] * idx_x + out_format->strides[2] * idx_y) / sizeof(float);
    if(idx_x<out_format->shape[1] && idx_y < out_format->shape[2])
        out[out_idx] = ans;
}


namespace opr{

Tensor matmul(const Tensor &x, const Tensor &y) {
    vector<size_t> x_shape = x.m_shape;
    vector<size_t> y_shape = y.m_shape;
    vector<size_t> x_strides = x.m_strides;
    vector<size_t> y_strides = y.m_strides;

    bool x_extended = false;
    if(x_shape.size() == 2) {
        x_shape.insert(x_shape.begin(), 1);
        x_strides.insert(x_strides.begin(), x_strides[0]);
        x_extended = true;
    }
    bool y_extended = false;
    if(y_shape.size() == 2) {
        y_shape.insert(y_shape.begin(), 1);
        y_strides.insert(y_strides.begin(), y_strides[0]);
        y_extended = true;
    }
    assert(x_shape[2] == y_shape[1]);
    
    
    vector<size_t> output_shape;
    vector<size_t> output_strides;
    size_t output_size = sizeof(float);
    output_shape.push_back(max(x_shape[0], y_shape[0]));
    output_shape.push_back(x_shape[1]);
    output_shape.push_back(y_shape[2]);

    for(size_t i=0;i<output_shape.size();i++) {
        output_strides.push_back(output_size);
        output_size *= output_shape[i];
    }
    float *res;
    cudaMalloc(&res, sizeof(float) * output_size);

    TensorFormat *x_format = TensorFormat::make_cuda_tensor_format(x_shape, x_strides);
    TensorFormat *y_format = TensorFormat::make_cuda_tensor_format(y_shape, y_strides);
    TensorFormat *out_format = TensorFormat::make_cuda_tensor_format(output_shape, output_strides);

    // 
    int block_size = 32;
    dim3 threads(block_size, block_size);
    dim3 blocks((output_shape[1] + block_size - 1)/block_size, 
            (output_shape[2] +block_size - 1)/block_size, 
            output_shape[0]);
    
    kernel_matmul<<<blocks, threads>>>(res, out_format, x.m_data.get(), x_format, y.m_data.get(), y_format);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error\n");
    }
    x_format->release();
    y_format->release();
    out_format->release();
    if(x_extended && y_extended) {
        output_shape.erase(output_shape.begin());
        output_strides.erase(output_strides.begin());
    }
    return Tensor(res, output_shape, output_strides);
}

}
