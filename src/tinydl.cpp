#include "tensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "opr/elementwise.h"
#include "opr/reduction.h"
#include "opr/matmul.h"
#include "opr/trans_layout.h"
#include "opr/cudnn/conv.h"
#include "opr/cudnn/batchnorm.h"


PYBIND11_MODULE(_tinydl, m) {
    m.doc() = "pybind11 simple example";

    m.def("op_add", &opr::add);
    m.def("op_sub", &opr::sub);
    m.def("op_mul", &opr::mul);
    m.def("op_div", &opr::div);
    
    m.def("op_equal", &opr::equal);
    m.def("op_less_then", &opr::less_then);
    m.def("op_less_equal", &opr::less_equal);
    m.def("op_greater_then", &opr::greater_then);
    m.def("op_greater_equal", &opr::greater_equal);

    // type convertion
    m.def("op_as_float32", &opr::as_float32);
    m.def("op_as_bool", &opr::as_bool);

    m.def("op_relu", &opr::relu);
    m.def("op_log", &opr::log);
    m.def("op_exp", &opr::exp);
    m.def("op_sigmoid", &opr::sigmoid);
    m.def("op_neg", &opr::neg);

    m.def("op_matmul", &opr::matmul);

    m.def("op_conv2d", &opr::conv2d);
    m.def("op_batchnorm", &opr::batchnorm);

    m.def("op_view", &opr::view,
            py::arg("tensor"), py::arg("axis"));

    m.def("op_reduce_sum", &opr::reduce_sum, 
            py::arg("input"), py::arg("axis"), py::arg("keep_dim")=false);
    m.def("op_reduce_mean", &opr::reduce_mean,
            py::arg("input"), py::arg("axis"), py::arg("keep_dim")=false);
    m.def("op_reduce_min", &opr::reduce_min,
            py::arg("input"), py::arg("axis"), py::arg("keep_dim")=false);
    m.def("op_reduce_max", &opr::reduce_max,
            py::arg("input"), py::arg("axis"), py::arg("keep_dim")=false);

    py::enum_<DataType>(m, "DataType")
        .value("float32", DataType::Float32)
        .value("uint64", DataType::UInt64)
        .value("int64", DataType::Int64)
        .value("bool", DataType::Bool)
        .export_values();

    py::class_ <Tensor, shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<py::array_t<float> &>())
        .def(py::init<py::array_t<uint64_t> &>())
        .def(py::init<py::array_t<int64_t> &>())
        .def(py::init<py::array_t<bool> &>())
        .def("requires_grad_", &Tensor::set_requires_grad)
        .def("zero_grad", &Tensor::zero_grad)
        .def("backward", &Tensor::backward)
        .def("grad_fn", &Tensor::grad_fn)
        .def("grad", &Tensor::grad)
        .def("graph_node", &Tensor::graph_node)
        .def("_to_numpy_float", &Tensor::to_numpy<float>)
        .def("_to_numpy_uint64", &Tensor::to_numpy<uint64_t>)
        .def("_to_numpy_int64", &Tensor::to_numpy<int64_t>)
        .def("_to_numpy_bool", &Tensor::to_numpy<bool>)
        .def("set_value", &Tensor::set_value)
        .def("dtype", &Tensor::dtype)
        .def_property_readonly("requires_grad", &Tensor::requires_grad)
        .def_property_readonly("id", &Tensor::id);

    py::class_<BackwardFunc, shared_ptr<BackwardFunc>>(m, "BackwardFunc");
    py::class_<GraphNode, shared_ptr<GraphNode>>(m, "GraphNode");
}
