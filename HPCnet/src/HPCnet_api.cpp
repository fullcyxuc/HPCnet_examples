#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "get_hausdorff_dis_gpu.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_hausdorff_dis_wrapper", &get_hausdorff_dis_wrapper_fast, "get_hausdorff_dis_wrapper_fast");
}
