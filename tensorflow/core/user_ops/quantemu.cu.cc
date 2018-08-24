#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "quantemu.h"
//#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"


using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;


#include "quantemu_cuda_utils.cu.cc" 
#if 0
template<int threads>
extern __global__ void find_min_max_dynamic(const float* in, float* out, int n, int start_adr, int num_blocks);

template<int els_per_block, int threads>
extern __global__ void find_min_max(const float* in, float* out);
#endif 

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

/* Define the GPU implementation that launches the CUDA kernel. */
template <typename T>
struct QuantEmuFunctor<GPUDevice, T> { 
  void operator()(const GPUDevice& d, int mbits, int rmode, int size, const T* in, T* out) {
    int thread_per_block = (32 < size)?32 :size;
    int block_count = size/thread_per_block; 

    find_min_max<32, 32><<<block_count, thread_per_block>>>(in, out);
     
    T absmax = 0;
    QuantEmuCudaKernel<T>
         <<<block_count, thread_per_block, 0, d.stream()>>>(mbits, absmax, rmode, size, in, out);
  }
};
template struct QuantEmuFunctor<GPUDevice, float>;

template <typename T>
struct BlockC_QuantEmuFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int mbits, int block_size, int rmode, const Tensor in, Tensor out) {
//    std::cout << " Inside the BlockQuantEmuFunctorGPU version " << std::endl; 
  }
};
template struct BlockC_QuantEmuFunctor<GPUDevice, float>;

template <typename T>
struct BlockCHW_QuantEmuFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int mbits, int block_size, int rmode, const Tensor in, Tensor out) {
//    std::cout << " Inside the BlockQuantEmuFunctorGPU version " << std::endl; 
  }
};
template struct BlockCHW_QuantEmuFunctor<GPUDevice, float>;

template <typename T>
struct LowpFloatQuantEmuFunctor <GPUDevice, T> {
  void operator()(const GPUDevice& d, int mbits, int exp_bits, int rmode, int size, const T* in, T* out) {
    std::cout << " Inside the LowpFloatQuantEmuFunctor GPU version "<< size  << std::endl; 
//    int thread_per_block = (32 < size)? 32 : size;
//    int block_count = size/thread_per_block; 

    int num_blocks = size/32;
    int tail = size - num_blocks*32;
    int start_adr = size - tail;
 
    find_min_max<32, 32><<< num_blocks, 32>>>(in, out);
    find_min_max_dynamic<32><<< 1, 32>>>(in, out, size, start_adr, num_blocks);
  }
};
template struct LowpFloatQuantEmuFunctor<GPUDevice, float>;

#endif  // GOOGLE_CUDA
