#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/Dispatch.h>

namespace at::native {

namespace {
__global__ void any_early_exit_kernel(
  const bool* data,
  int64_t n,
  int* found
) {
  if (*found) return;

  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  bool val = (idx < n) && data[idx];

  if (__any_sync(0xffffffff, val)) {
    atomicOr(found, 1);
  }
}

void any_early_exit_cuda(const bool* data, int64_t n, bool* output) {
  int* found;
  cudaMalloc(&found, sizeof(int));
  cudaMemset(found, 0, sizeof(int));

  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  any_early_exit_kernel<<<grid_size, block_size>>>(data, n, found);

  int result;
  cudaMemcpy(&result, found, sizeof(int), cudaMemcpyDeviceToHost);
  *output = result != 0;

  cudaFree(found);
}

}

void and_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "and_cuda", [&]() {
        gpu_reduce_kernel<scalar_t, bool>(
            iter,
            func_wrapper<bool>([] GPU_LAMBDA(bool acc, scalar_t val) -> bool {
              return (acc && static_cast<bool>(val));
            }),
            true);
      });
}

void or_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == kBool && iter.is_contiguous()) {
    auto* data = static_cast<const bool*>(iter.data_ptr(1));
    auto* output = static_cast<bool*>(iter.data_ptr(0));
    any_early_exit_cuda(data, iter.numel(), output);
    return;
  }
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "or_cuda", [&]() {
        gpu_reduce_kernel<scalar_t, bool>(
            iter,
            func_wrapper<bool>([] GPU_LAMBDA(bool acc, scalar_t val) -> bool {
              return (acc || static_cast<bool>(val));
            }),
            false);
      });
}

REGISTER_DISPATCH(and_stub, &and_kernel_cuda)
REGISTER_DISPATCH(or_stub, &or_kernel_cuda)

} // namespace at::native
