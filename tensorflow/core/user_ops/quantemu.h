#ifndef _KERNEL_H_
#define _KERNEL_H_

#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

enum ROUND_MODE{NOROUND=0, BIASED, NEAREST};
enum FGQ_TYPE{NOBLOCK=0, BLOCK_C, BLOCK_CHW };
enum LPDATA_TYPE{DFP_INT=0, DFP_UINT, TERNARY, BLK_FP, LOWP_FP};

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename Device, typename T>
struct QuantEmuFunctor {
  void operator()(const Device& d, int mbits, T absmax, int rmode, int size, const T* in, T* out);
};

template <typename Device, typename T>
struct BlockQuantEmuFunctor {
  void operator()(const Device& d, int mbits, int rmode, const Tensor in, Tensor out);
};

template <typename Device, typename T>
struct LowpFloatQuantEmuFunctor {
  void operator()(const Device& d, int mbits, int exp_bits, int rmode, int size, const T *in, T *out);
};

template <typename Device, typename T>
struct BlockFloatQuantEmuFunctor {
  void operator()(const Device& d, int mbits, int rmode, const Tensor in, Tensor out);
};


#if 0 //GOOGLE_CUDA
template <typename GPUDevice, typename T>
struct QuantEmuFunctor {
  void operator()(const GPUDevice& d, int mbits, T absmax, int rmode, int size, const T* in, T* out);
};
#endif 

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

#endif //_KERNEL_H_
