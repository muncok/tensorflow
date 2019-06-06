#ifndef _KERNEL_H_
#define _KERNEL_H_

#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

enum ROUNDING_MODES{TRUNCATE=0, ROUND_RNE=1, ROUND_STOCHASTIC=2};
enum FGQ_TYPE{NOBLOCK=0, BLOCK_C, BLOCK_CHW };
enum LPDATA_TYPE{INT=1, UINT=2, LOWP_FP=3, LOG2=4, POSIT=5, BFLOAT=6, MODFP16=7, BLOCK_FP=8};

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
  void operator()(const Device& d, int unsigned_data, int mbits, int rmode, int size, T* in, T* out);
};

template <typename Device, typename T>
struct BlockC_QuantEmuFunctor {
  void operator()(const Device& d, int unsigned_data, int mbits, int *dims, int block_size, int rmode, T* in, T* out);
};

template <typename Device, typename T>
struct BlockCHW_QuantEmuFunctor {
  void operator()(const Device& d, int unsigned_data, int mbits, int *dims, int block_size, int rmode, T *in, T *out);
};

template <typename Device, typename T>
struct LowpFloatQuantEmuFunctor {
  void operator()(const Device& d, int mbits, int exp_bits, int rmode, int size, const T *in, T *out);
};

template <typename Device, typename T>
struct Log2QuantEmuFunctor {
  void operator()(const Device& d, int size, const T *in, T *out);
};

template <typename Device, typename T>
struct BlockFloatQuantEmuFunctor {
  void operator()(const Device& d, int mbits, int rmode, const Tensor in, Tensor out);
};

template <typename Device, typename T>
struct PositQuantEmuFunctor {
  void operator()(const Device& d, int m_bits, int es_bits, int rmode, int size, const T *in, T *out);
};
template <typename Device, typename T>
struct BfloatQuantEmuFunctor {
  void operator()(const Device& d, int rmode, int size, const T *in, T *out);
};
template <typename Device, typename T>
struct ModFP16QuantEmuFunctor{
  void operator()(const Device& d, int size, const T *in, T *out);
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
