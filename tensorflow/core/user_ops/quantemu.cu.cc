#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "quantemu.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void QuantEmuCudaKernel(int mbits, T absmax, int rmode, const int size, const T* in, T* out) {
  int quant_max = pow(2, mbits-1) - 1;
  T sfquant = (T)(quant_max/absmax);
  T sfdequant = (T)1.0/sfquant;
  T sfquant_br = sfquant * 4.0;  /* quantize to nbits + 2, we need them for rounding */
 
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
      T inval = in[i];
      /* saturate anything larger than fmax_val to fmax_val */
      if (inval > absmax) inval = absmax;
      int ival = (int)(inval * sfquant);
      int rbias = ((int)(inval * sfquant_br)) & 0x3;
      //if (in[i] == 0.0) { ival = 0; rbias = 0; }
      int negative = (ival < 0);
      /* disable sign bit for rounding */
      if(negative) ival = 0 - ival;
      ival += ((rmode==BIASED) & (rbias > 0));
      ival += ((rmode==NEAREST) & (rbias > 1));
      /* saturate after rounding */
      if (ival > quant_max) ival = quant_max;
      /* restore sign */
      if(negative) ival = 0 - ival;
      out[i] = ival * sfdequant;
  }
}



// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct QuantEmuFunctor<GPUDevice, T> { 
  void operator()(const GPUDevice& d, int mbits, T absmax, int rmode, int size, const T* in, T* out) {
  /* Launch the cuda kernel.
     See core/util/cuda_kernel_helper.h for example of computing
     block count and thread_per_block count.
  */
    std::cout << "Calling Cuda kernel " << std::endl;
#if 10
    int block_count = 1024;
    int thread_per_block = 20;
    QuantEmuCudaKernel<T>
         <<<block_count, thread_per_block, 0, d.stream()>>>(mbits, absmax, rmode, size, in, out);
#endif 
  }
};

/* Explicitly instantiate functors for the types of OpKernels registered.*/
template struct QuantEmuFunctor<GPUDevice, float>;
//template struct QuantEmuFunctor<GPUDevice, int32>;

template <typename T>
struct BlockQuantEmuFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int mbits, int rmode, const Tensor in, Tensor out) {
  }
};
/* Explicitly instantiate functors for the types of OpKernels registered.*/
template struct BlockQuantEmuFunctor<GPUDevice, float>;
//template struct BlockQuantEmuFunctor<GPUDevice, int32>;

template <typename T>
struct LowpFloatQuantEmuFunctor <GPUDevice, T> {
  void operator()(const GPUDevice& d, int mbits, int exp_bits, int rmode, int size, const T* in, T* out) {
  }
};
template struct LowpFloatQuantEmuFunctor<GPUDevice, float>;
//template struct LowpFloatQuantEmuFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA

