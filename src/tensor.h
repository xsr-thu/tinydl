#ifndef __TENSOR_H_
#define __TENSOR_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <memory>

namespace py = pybind11;
using namespace std;

struct Tensor {
    shared_ptr<float> m_data;
    size_t m_size;
    vector<size_t> m_shape;
    vector<size_t> m_strides;

    Tensor(py::array_t<float> arr);

    Tensor(float *data, vector<size_t> shape, vector<size_t> strides);

    ~Tensor();

    size_t size();

    size_t dim();

    py::array_t<float> to_numpy();
};




#endif
