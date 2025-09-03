#include "tensor.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
PYBIND11_MODULE(ops, m) {
    m.doc() = "cuda ops backend"; // optional module docstring

    m.def("free_tensor", &free_tensor, "A function that free a tensor on GPU given the pointer");
    m.def("copy_tensor_d2h", &copy_tensor_d2h, "A function that copy a tensor from device to host given the pointer and number of elements");
    m.def("copy_tensor_h2d", &copy_tensor_h2d, "A function that copy a tensor from host to device given the pointer and number of elements");
}