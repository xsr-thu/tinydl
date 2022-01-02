#include "tensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "opr/elementwise.h"
#include "opr/reduction.h"
#include "opr/matmul.h"


PYBIND11_MODULE(tinydl, m) {
    m.doc() = "pybind11 simple example";

    m.def("op_add", &opr::add);
    m.def("op_sub", &opr::sub);
    m.def("op_mul", &opr::mul);
    m.def("op_div", &opr::div);
    
    m.def("op_relu", &opr::relu);
    m.def("op_log", &opr::log);
    m.def("op_exp", &opr::exp);
    m.def("op_sigmoid", &opr::sigmoid);
    
    m.def("op_matmul", &opr::matmul);

    m.def("op_reduce_sum", &opr::reduce_sum, 
            py::arg("input"), py::arg("axis"), py::arg("keep_dim")=false);
    m.def("op_reduce_mean", &opr::reduce_mean,
            py::arg("input"), py::arg("axis"), py::arg("keep_dim")=false);

    py::class_ <Tensor>(m, "Tensor")
        .def(py::init<py::array_t<float> &>())
        .def("to_numpy", &Tensor::to_numpy);
}
