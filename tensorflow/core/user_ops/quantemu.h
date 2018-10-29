#ifndef _KERNEL_H_
#define _KERNEL_H_

#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

enum ROUND_MODE{NOROUND=0, BIASED, NEAREST};
enum FGQ_TYPE{NOBLOCK=0, BLOCK_C, BLOCK_CHW };
enum LPDATA_TYPE{DFP_INT=1, LOWP_FP, POSIT, BLOCK_FP};

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

typedef union ufloat32
{
  unsigned u;
  float f;
  struct
  {
    uint Mantissa : 23;
    uint Exponent : 8;
    uint Sign : 1;
  };
}UFLOAT32;

template <typename Device, typename T>
struct QuantEmuFunctor {
  void operator()(const Device& d, int unsigned_data, int mbits, int rmode, int size, const T* in, T* out);
};

template <typename Device, typename T>
struct BlockC_QuantEmuFunctor {
  void operator()(const Device& d, int unsigned_data, int mbits, int *dims, int block_size, int rmode, const T* in, T* out);
};

template <typename Device, typename T>
struct BlockCHW_QuantEmuFunctor {
  void operator()(const Device& d, int unsigned_data, int mbits, int *dims, int block_size, int rmode, const T *in, T *out);
};

template <typename Device, typename T>
struct LowpFloatQuantEmuFunctor {
  void operator()(const Device& d, int mbits, int exp_bits, int rmode, int size, const T *in, T *out);
};

template <typename Device, typename T>
struct BlockFloatQuantEmuFunctor {
  void operator()(const Device& d, int mbits, int rmode, const Tensor in, Tensor out);
};

template <typename Device, typename T>
struct PositQuantEmuFunctor {
  void operator()(const Device& d, int m_bits, int es_bits, int size, const T *in, T *out);
};

#if GOOGLE_CUDA
#if 0
template <typename Eigen::GpuDevice, typename T>
struct QuantEmuFunctor {
  void operator()(const Eigen::GpuDevice& d, int mbits, int rmode, int size, const T* in, T* out);
};

template <typename Eigen::GpuDevice, typename T>
struct LowpFloatQuantEmuFunctor {
  void operator()(const Eigen::GpuDevice& d, int mbits, int exp_bits, int rmode, int size, const T *in, T *out);
};

template <typename Eigen::GpuDevice, typename T>
struct BlockFloatQuantEmuFunctor {
  void operator()(const Eigen::GpuDevice& d, int mbits, int rmode, const Tensor in, Tensor out);
};

template <typename T>
void QuantEmuFunctor_GPU_launcher (const GPUDevice& d, int mbits, T absmax, int rmode, int size, const T* in, T* out); 
template <typename T>
void BlockQuantEmuFunctor_GPU_launcher (const GPUDevice& d, int mbits, int rmode, const Tensor in, Tensor out); 
template <typename T>
void LowpFloatQuantEmuFunctor_GPU_launcher (const GPUDevice& d, int mbits, int exp_bits, int rmode, int size, const T* in, T* out); 
#endif 

#endif 

#endif //_KERNEL_H_
