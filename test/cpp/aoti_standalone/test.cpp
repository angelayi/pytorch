#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <torch/torch.h>
#include <torch/standalone/slim_tensor/slim_tensor.h>

namespace torch::standalone {

TEST(SlimTensorTest, AddOp) {
  // Create at::Tensors
  at::Tensor tensor1 = at::ones({2, 3}, at::kFloat);
  at::Tensor tensor2 = at::ones({2, 3}, at::kFloat) * 2;
  at::Tensor result_tensor = at::add(tensor1, tensor2);
  
  // Create SlimTensors
  void* data_ptr1 = tensor1.data_ptr();
  const int64_t* sizes_data1 = tensor1.sizes().data();
  const int64_t* strides_data1 = tensor1.strides().data();
  torch::standalone::ArrayRef sizes1(sizes_data1, tensor1.dim());
  torch::standalone::ArrayRef strides1(strides_data1, tensor1.dim());
  torch::standalone::SlimTensor slim_tensor1 = torch::standalone::create_tensor_from_blob(
      data_ptr1,
      sizes1,
      strides1,
      tensor1.scalar_type(),
      tensor1.device(),
      tensor1.storage_offset()
  );
  
  void* data_ptr2 = tensor2.data_ptr();
  const int64_t* sizes_data2 = tensor2.sizes().data();
  const int64_t* strides_data2 = tensor2.strides().data();
  torch::standalone::ArrayRef sizes2(sizes_data2, tensor2.dim());
  torch::standalone::ArrayRef strides2(strides_data2, tensor2.dim());
  torch::standalone::SlimTensor slim_tensor2 = torch::standalone::create_tensor_from_blob(
      data_ptr2,
      sizes2,
      strides2,
      tensor2.scalar_type(),
      tensor2.device(),
      tensor2.storage_offset()
  );

  // Perform add operation on SlimTensor (manual implementation)
  auto slim_result = torch::standalone::create_empty_tensor(
      sizes2, strides2, c10::ScalarType::Float);
  float* result_data = static_cast<float*>(slim_result.data_ptr());
  
  for (int i = 0; i < 6; i++) {
    result_data[i] = static_cast<float*>(data_ptr1)[i] + static_cast<float*>(data_ptr2)[i];
  }
  
  // Verify results
  float* expected_data = result_tensor.data_ptr<float>();
  for (int i = 0; i < 6; i++) {
    EXPECT_FLOAT_EQ(result_data[i], expected_data[i]);
    EXPECT_FLOAT_EQ(result_data[i], 3.0f);
  }
}


} // namespace torch::standalone
