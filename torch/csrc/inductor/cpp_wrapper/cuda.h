#pragma once

#ifdef AOTI_LIBTORCH_FREE
#include <torch/csrc/inductor/aoti_standalone/cuda/c_shim_cuda.h>
#include <torch/csrc/inductor/aoti_standalone/cuda/utils.h>
#else

#include <torch/csrc/inductor/cpp_wrapper/common.h>
#include <torch/csrc/inductor/cpp_wrapper/device_internal/cuda.h>
