#ifndef __TENSOR_H_
#define __TENSOR_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <memory>

namespace py = pybind11;
using namespace std;




struct TensorStorage {
    float* m_data;
    size_t m_size;
    vector<size_t> m_shape;
    vector<size_t> m_strides;

    TensorStorage(float* data, size_t size, vector<size_t> shape, vector<size_t> strides)
    : m_data(data), m_size(size), m_shape(shape), m_strides(strides) {
    }


    ~TensorStorage() {
        if(m_data)
            cudaFree(m_data);
    }
};


struct Tensor {
    shared_ptr<TensorStorage> m_storage;

    Tensor(py::array_t<float> arr);

    Tensor(float *data, vector<size_t> shape, vector<size_t> strides);

    ~Tensor();

    size_t size();

    size_t dim();

    py::array_t<float> to_numpy();
};




#endif
