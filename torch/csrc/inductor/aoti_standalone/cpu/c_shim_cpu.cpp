#include <torch/csrc/inductor/aoti_standalone/cpu/c_shim_cpu.h>

AOTITorchError aoti_torch_cpu_addmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat1,
    AtenTensorHandle mat2,
    double beta,
    double alpha) {
  return AOTI_TORCH_SUCCESS;
}
