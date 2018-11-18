#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "quantemu.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
//#include "posit_pack.h" 

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

#define CUBLOCK_SIZE 512 

#include "posit_impl.cu.cc" 

__device__ 
Eigen::half atomicMinf(
	Eigen::half *address_half, 
	float val)
{
    float address = __half2float (*address_half);

    int *address_as_int =(int*)&address;
    int old = *address_as_int, assumed;
    while (val < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
        }
    float fin = __int_as_float(old);
    return __float2half_rn (fin); 
}

__device__ 
Eigen::half atomicMaxf(
	Eigen::half *address_half, 
	float val)
{
    float address = __half2float (*address_half);

    int *address_as_int =(int*)&address;
    int old = *address_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
        }
    float fin = __int_as_float(old);
    return __float2half_rn (fin); 
}

__device__ 
float atomicMinf(
	float* address, 
	float val)
{
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
        }
    return __int_as_float(old);
}

__device__ 
float atomicMaxf(
	float* address, 
	float val)
{
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
        }
    return __int_as_float(old);
}

__global__ 
void max_reduce(
	const Eigen::half* 
	const d_array, 
	Eigen::half *d_max, 
	const int block_offset, 
	const int block_size)
{
#if 0
    extern __shared__ float shared[]; 
#else 
    volatile __shared__ float shared[CUBLOCK_SIZE]; 
#endif 
    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid + block_offset; 
    shared[tid] = 0; 
    size_t elements = block_offset+ block_size; 

    while (gid < elements) {
        shared[tid] = fmaxf(fabsf(shared[tid]), fabsf(__half2float(d_array[gid])));
        gid += gridDim.x*blockDim.x;
        }
    __syncthreads();
    gid = (blockDim.x * blockIdx.x) + tid + block_offset;  // 1
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s && gid < elements)
//        if (tid < s)
            shared[tid] = fmaxf(fabsf(shared[tid]), fabsf(shared[tid + s]));
        __syncthreads();
    }

    if (tid == 0)
      atomicMaxf(d_max, shared[0]);
}

__global__ 
void max_reduce(
	const float* const d_array, 
	float* d_max, 
	const int block_offset, 
	const int block_size)
{
#if 0 
    extern __shared__ float shared[]; 
#else 
    volatile __shared__ float shared[CUBLOCK_SIZE]; 
#endif 
    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid + block_offset; 
    shared[tid] = 0; 
    size_t elements = block_offset+ block_size; 

    while (gid < elements) {
        shared[tid] = fmaxf(fabsf(shared[tid]), fabsf(d_array[gid]));
        gid += gridDim.x*blockDim.x;
        }
    __syncthreads();
    gid = (blockDim.x * blockIdx.x) + tid + block_offset;  // 1
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s && gid < elements)
//        if (tid < s )
            shared[tid] = fmaxf(fabsf(shared[tid]), fabsf(shared[tid + s]));
        __syncthreads();
    }

    if (tid == 0)
      atomicMaxf(d_max, shared[0]);
}

__global__ 
void min_max_reduce(
	const Eigen::half* 
	const d_array, 
	Eigen::half *d_min, 
	Eigen::half *d_max, 
	const int block_offset, 
	const int block_size)
{
#if 0
    extern __shared__ float shared[]; 
    volatile float *shared_max = &shared[0]; 
    volatile float *shared_min = &shared[CUBLOCK_SIZE]; 
#else 
    volatile __shared__ float shared_max[CUBLOCK_SIZE]; 
    volatile __shared__ float shared_min[CUBLOCK_SIZE]; 
#endif 

    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid + block_offset; 
    shared_max[tid] = 0; 
    shared_min[tid] = 0; 
    size_t elements = block_offset+ block_size; 

    while (gid < elements) {
        shared_max[tid] = fmaxf(shared_max[tid], __half2float(d_array[gid]));
        shared_min[tid] = fminf(shared_min[tid], __half2float(d_array[gid]));
        gid += gridDim.x*blockDim.x;
        }
    __syncthreads();
    gid = (blockDim.x * blockIdx.x) + tid + block_offset;  // 1
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s && gid < elements)
//        if (tid < s)
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
            shared_min[tid] = fminf(shared_min[tid], shared_min[tid + s]);
        __syncthreads();
    }

    __syncthreads();
    if (tid == 0) {
      atomicMaxf(d_max, shared_max[0]);
      atomicMinf(d_min, shared_min[0]);
    }
}

__global__ 
void min_max_reduce(
	const float* const d_array, 
	float* d_min, 
	float* d_max, 
	const int block_offset, 
	const int block_size)
{
#if 0
    extern __shared__ float shared[]; 
    volatile float *shared_max = &shared[0]; 
    volatile float *shared_min = &shared[CUBLOCK_SIZE]; 
#else 
    volatile __shared__ float shared_max[CUBLOCK_SIZE]; 
    volatile __shared__ float shared_min[CUBLOCK_SIZE]; 
#endif 

    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid + block_offset; 
    shared_max[tid] = 0; 
    size_t elements = block_offset+ block_size; 

    while (gid < elements) {
        shared_max[tid] = fmaxf(shared_max[tid], d_array[gid]);
        shared_min[tid] = fminf(shared_min[tid], d_array[gid]);
        gid += gridDim.x*blockDim.x;
    }
    __syncthreads();
    gid = (blockDim.x * blockIdx.x) + tid + block_offset;  // 1
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s && gid < elements)
//        if (tid < s )
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
            shared_min[tid] = fminf(shared_min[tid], shared_min[tid + s]);
        __syncthreads();
    }

    __syncthreads();
    if (tid == 0) {
      atomicMaxf(d_max, shared_max[0]);
      atomicMinf(d_min, shared_min[0]);
    }
}

__global__ 
void reduce_interquartile_sum (
	const float* const d_array, 
	float *d_sum, 
	float *d_min, 
	float *d_max, 
	const int block_offset, 
	const int block_size)
{
#if 0 
    extern __shared__ float shared[]; 
#else 
    volatile __shared__ float shared[CUBLOCK_SIZE]; 
#endif 
    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid + block_offset; 
    shared[tid] = 0; 

    size_t elements = block_offset+ block_size; 
    /* interquanrtile region is between 25% and 75% of the probability distribution */ 
    float iqmax = *d_max * (float) 0.75; 
    float iqmin = *d_min * (float) 0.75; 

    while (gid < elements) {
        shared[tid] += (d_array[gid] < iqmax && d_array[gid] > iqmin) ? d_array[gid] : 0;
        gid += gridDim.x*blockDim.x;
    }
    __syncthreads();
    gid = (blockDim.x * blockIdx.x) + tid + block_offset;  // 1
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s && gid < elements)
//        if (tid < s )
            shared[tid] += shared[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
      atomicAdd(d_sum, shared[0]);
    }
}

__global__ 
void reduce_interquartile_sum(
	const Eigen::half* const d_array, 
	float *d_sum, 
	Eigen::half *d_min, 
	Eigen::half *d_max, 
	const int block_offset, 
	const int block_size)
{
#if 0
    extern __shared__ float shared[]; 
#else 
    volatile __shared__ float shared[CUBLOCK_SIZE]; 
#endif 
    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid + block_offset; 
    shared[tid] = 0; 
    size_t elements = block_offset+ block_size; 
    /* interquanrtile region is between 25% and 75% of the probability distribution */ 
    Eigen::half iqmax = *d_max * (Eigen::half) 0.75; 
    Eigen::half iqmin = *d_min * (Eigen::half) 0.75; 

    while (gid < elements) {
        shared[tid] += (d_array[gid] < iqmax && d_array[gid] > iqmin) ? __half2float(d_array[gid]) : 0;
        gid += gridDim.x*blockDim.x;
        }
    __syncthreads();
    gid = (blockDim.x * blockIdx.x) + tid + block_offset;  // 1
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s && gid < elements)
//        if (tid < s )
            shared[tid] += shared[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
      atomicAdd(d_sum, shared[0]);
    }
}

__global__ 
void estimate_mode_range (
        float *d_absmax, 
        float *d_sum, 
   	float *d_min, 
	float *d_max, 
	int block_size) 
{
//    float ninty_perc = (float)0.9f; 
//    float seventy_perc = (float)0.7f; 

    __syncthreads();

    float iq_mean = *d_sum /(float)block_size;  
    if (threadIdx.x == 0)
    { 
      	if (*d_min < (float)0.0 && *d_max > (float)0.0) {
           /* uniform distribution */  
           if (fabsf(*d_min - iq_mean) < fabsf(*d_max - iq_mean)) *d_absmax = (float)0.9*fabsf(*d_min);  
           else *d_absmax = (float)0.9*fabsf(*d_max);
      	} else if (*d_min < (float)0.0 && *d_max < (float)0.0) {
           if (fabsf(*d_min - iq_mean) < fabsf(*d_max - iq_mean)) *d_absmax = (float)0.9*fabsf(*d_min);  
           else *d_absmax = (float)0.9*fabsf(*d_min);
      	} else if (*d_min > (float)0.0 && *d_max > (float)0.0) {
           if (fabsf(*d_min - iq_mean) < fabsf(*d_max - iq_mean)) *d_absmax = (float)0.9*fabsf(*d_max);  
           else *d_absmax = (float)0.9*fabsf(*d_max);
      	} 
    }
    __syncthreads();
}

__global__ 
void estimate_mode_range (
        Eigen::half *d_absmax, 
        float *d_sum, 
   	Eigen::half *d_hmin, 
	Eigen::half *d_hmax, 
	int block_size) 
{

    __syncthreads();

    float iq_mean = *d_sum /(float)block_size;  
    float d_min = __half2float(*d_hmin); 
    float d_max = __half2float(*d_hmax); 

    if (threadIdx.x == 0)
    { 
      	if (d_min < (float)0.0 && d_max > (float)0.0) {
       	/* uniform distribution */  
           if (fabsf(d_min - iq_mean) < fabsf(d_max - iq_mean)) *d_absmax = __float2half_rn((float)0.9*fabsf(d_min));  
           else *d_absmax = __float2half_rn((float)0.9*fabsf(d_max));
      	} else if (d_min < (float)0.0 && d_max < (float)0.0) {
           if (fabsf(d_min - iq_mean) < fabsf(d_max - iq_mean)) *d_absmax = __float2half_rn((float)0.9*fabsf(d_min));  
           else *d_absmax = __float2half_rn((float)0.9*fabsf(d_min));
      	} else if (d_min > (float)0.0 && d_max > (float)0.0) {
           if (fabsf(d_min - iq_mean) < fabsf(d_max - iq_mean)) *d_absmax = __float2half_rn((float)0.9*fabsf(d_max));  
           else *d_absmax = __float2half_rn((float)0.9*fabsf(d_max));
      	} 
    }
    __syncthreads();
}

__global__ 
void QuantEmuCudaKernel(
	int unsigned_data, 
	int mbits, 
	Eigen::half *absmax, 
	int rmode, 
	const int block_offset, 
	const int block_size, 
	Eigen::half *in, 
	Eigen::half *out) 
{
  int quant_max; 
  float sfquant;
  float sfdequant;

  quant_max = pow(2, mbits-1) - 1;
  if (unsigned_data) quant_max = pow(2, mbits) - 1;
  sfquant = (float)(quant_max / __half2float(*absmax));
  sfdequant = (float)1.0/sfquant;
//  float sfquant_br; 
//  sfquant_br = sfquant * 4.0;  /* quantize to nbits + 2, we need them for rounding */ 

  int tid = threadIdx.x;
  int gid = (blockDim.x * blockIdx.x) + tid + block_offset;
  int size = block_offset + block_size; 
  float fabsfmax = __half2float(*absmax); 
 
//  for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x + block_offset; gid < size; gid += blockDim.x * gridDim.x) {
  for (gid; gid < size; gid += blockDim.x*gridDim.x) {
      float inval = __half2float(in[gid]);
      /* saturate anything larger than fmax_val to fmax_val */
      inval = fminf (inval, fabsfmax); 
      inval = fmaxf (inval, -fabsfmax);
      /* write back saturated value */ 
      in[gid] = __float2half_rn(inval);

      //int ival = (int)(inval * sfquant);
      /* round to nearest even */ 
      int ival = __half2int_rn (inval * sfquant);
#if 0
      int rbias = ((int)(inval * sfquant_br)) & 0x3;
      if (in[gid] == 0.0) { ival = 0; rbias = 0; }
      int negative = (ival < 0);
      /* disable sign bit for rounding */
      if(negative) ival = 0 - ival;
      ival += ((rmode==BIASED) & (rbias > 0));
      ival += ((rmode==NEAREST) & (rbias > 1));
      /* saturate after rounding */
      if (ival > quant_max) ival = quant_max;
      /* restore sign */
      if(negative) ival = 0 - ival;
#endif 
      out[gid] = __float2half_rn (ival * sfdequant);
  }
}

__global__ 
void QuantEmuCudaKernel(
        int unsigned_data, 
	int mbits, 
	float *absmax, 
	int rmode, 
	const int block_offset, 
	const int block_size, 
	float* in, float* out) 
{
  int quant_max; 
  float sfquant;
  float sfdequant;

  quant_max = pow(2, mbits-1) - 1;
  if (unsigned_data == 1) quant_max = pow(2, mbits) - 1;
  sfquant = (float)(quant_max / *absmax);
  sfdequant = (float)1.0/sfquant;
//  float sfquant_br; 
//  sfquant_br = sfquant * 4.0;  /* quantize to nbits + 2, we need them for rounding */ 

  int tid = threadIdx.x;
  int gid = (blockDim.x * blockIdx.x) + tid + block_offset;
  int size = block_offset + block_size; 
  float fabsfmax = *absmax; 
 
  //for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x + block_offset; gid < size; gid += blockDim.x * gridDim.x) {
  for (gid; gid < size; gid += blockDim.x*gridDim.x) {
      float inval = in[gid];
      /* saturate anything larger than fmax_val to fmax_val */
      inval = fminf (inval, fabsfmax); 
      inval = fmaxf (inval, -fabsfmax);
      /* write back saturated value */ 
      in[gid] = inval; 
#if 10
      /* round to the nearest even */ 
      int ival = __float2int_rn (inval * sfquant);
#else 
      int ival = (int)(inval * sfquant);
      int rbias = ((int)(inval * sfquant_br)) & 0x3;
      if (in[gid] == 0.0) { ival = 0; rbias = 0; }
      int negative = (ival < 0);
      /* disable sign bit for rounding */
      if(negative) ival = 0 - ival;
      ival += ((rmode==BIASED) & (rbias > 0));
      ival += ((rmode==NEAREST) & (rbias > 1));
      /* saturate after rounding */
      if (ival > quant_max) ival = quant_max;
      /* restore sign */
      if(negative) ival = 0 - ival;
#endif 
      out[gid] = ival * sfdequant;
  }
}

typedef union half_t { 
   unsigned short u; 
   __half f; 
} __half_t; 

__global__ 
void QuantEmuLowpCudaKernel(
	int mbits, 
	int exp_bits, 
	int rmode, 
	const int size, 
	const float *in, 
	float *out) 
{
  int non_mant_bits = exp_bits + 1; /* exponent + sign */
  int shift = 10 - (mbits - non_mant_bits);
  unsigned short lowpfp_mask = (unsigned short)(0xFFFF << shift);

  for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size; gid += blockDim.x * gridDim.x) {
      __half_t h; 
      float inval = in[gid];
      __half  hval = __float2half_rn(inval); 
      h.f = hval;
      h.u = (h.u & lowpfp_mask); 
      float outval = __half2float(h.f);
      out[gid] = outval;
  }
}

__global__ 
void QuantEmuLowpCudaKernel(
	int mbits, 
	int exp_bits, 
	int rmode, 
	const int size, 
	const Eigen::half* in, 
	Eigen::half* out) 
{
  int non_mant_bits = exp_bits + 1; /* exponent + sign */
  int shift = 10 - (mbits - non_mant_bits);
  unsigned short lowpfp_mask = (unsigned short)(0xFFFF << shift);

  for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size; gid += blockDim.x * gridDim.x) {
      __half_t h; 
      Eigen::half inval = in[gid];
      __half  hval = inval; 
      h.f = hval;
      h.u = (h.u & lowpfp_mask); 
      Eigen::half outval = h.f; 
      out[gid] = outval;
  }
}

__global__ 
void QuantEmuPositCudaKernel(
	int m_bits, 
	int es_bits, 
	const int size, 
	const float *in, 
	float *out) 
{
  for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size; gid += blockDim.x * gridDim.x) {
      float inval = in[gid];
      POSIT_UTYPE pval =_pack_posit(_unpack_float(inval), m_bits, es_bits); 
      float outval = _pack_float (_unpack_posit(pval, m_bits, es_bits));  
      out[gid] = outval;
  }
}

__global__ 
void QuantEmuPositCudaKernel(
	int m_bits, 
	int es_bits, 
	const int size, 
	const Eigen::half *in, 
	Eigen::half *out) 
{
  for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size; gid += blockDim.x * gridDim.x) {
      __half inval = in[gid];
      POSIT_UTYPE pval =_pack_posit(_unpack_float(__half2float(inval)), m_bits, es_bits); 
      float outval = _pack_float (_unpack_posit(pval, m_bits, es_bits));  
      out[gid] = __float2half_rn(outval);
  }
}

/* Define the GPU implementation that launches the CUDA kernel. */
template <typename T>
struct QuantEmuFunctor<GPUDevice, T> { 
  void operator()(const GPUDevice& d, int pruning_algo, int unsigned_data, int mbits, int rmode, int size, T* in, T* out) {
    int block = CUBLOCK_SIZE; 
    int grid = (size + (CUBLOCK_SIZE -1))/CUBLOCK_SIZE;  
    T *d_absmax;
    T *d_min;
    T *d_max;
    float *d_sum;
    cudaMalloc((void**) &d_absmax, sizeof(T));
    cudaMalloc((void**) &d_min, sizeof(T));
    cudaMalloc((void**) &d_max, sizeof(T));
    cudaMalloc((void**) &d_sum, sizeof(float));
    cudaMemset(&d_absmax, 0, sizeof(T));
    cudaMemset(&d_min, 0, sizeof(T));
    cudaMemset(&d_max, 0, sizeof(T));
    cudaMemset(&d_sum, 0, sizeof(float));
    if (pruning_algo == 0) { 
      max_reduce<<<grid, block, block*sizeof(float), d.stream()>>>(in, d_absmax, 0, size); 
    } else { 
      min_max_reduce<<<grid, block, 2*block*sizeof(float), d.stream()>>>(in, d_min, d_max, 0, size); 
      reduce_interquartile_sum<<<grid, block, block*sizeof(float), d.stream()>>>(in, d_sum, d_min, d_max, 0, size); 
      estimate_mode_range <<<grid, block, 0, d.stream()>>>(d_absmax, d_sum, d_min, d_max, size);
    }
    //cudaMemcpy(&absmax, d_absmax, sizeof(T), cudaMemcpyDeviceToHost); 
    QuantEmuCudaKernel <<<grid, block, 0, d.stream()>>>(unsigned_data, mbits, d_absmax, rmode, 0, size, in, out);
//    cudaStreamSynchronize(d.stream());
    cudaFree(d_absmax);
    cudaFree(d_min);
    cudaFree(d_max);
    cudaFree(d_sum);
  }
};
template struct QuantEmuFunctor<GPUDevice, float>;
template struct QuantEmuFunctor<GPUDevice, Eigen::half>;

template <typename T>
struct BlockC_QuantEmuFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int pruning_algo, int unsigned_data, int mbits, int *dims , 
	int block_size, int rmode, T *in, T *out) {

    int c_blocks =  dims[3]/block_size; 
    if ((dims[3]%block_size) || (dims[3] < block_size)) { c_blocks = 1; block_size = dims[3];}

    int block = CUBLOCK_SIZE; //(CUBLOCK_SIZE > block_size)?block_size:CUBLOCK_SIZE; 
    int grid = (block_size + (block -1))/block;  
    T *input_flat = in; 
    T *output_flat = out; 

    int num_cblocks = dims[0]*dims[1]*dims[2];
    const int num_streams = num_cblocks;  
    cudaStream_t streams[num_streams];
    T *d_absmax;
    T *d_min;
    T *d_max;
    float *d_sum;

    cudaMalloc((void**) &d_absmax, num_streams*sizeof(T));
    cudaMalloc((void**) &d_min, num_streams*sizeof(T));
    cudaMalloc((void**) &d_max, num_streams*sizeof(T));
    cudaMalloc((void**) &d_sum, num_streams*sizeof(float));
   
    for (int s =0; s < num_streams; s++ ) cudaStreamCreate(&streams[s]);

    for (int cb = 0; cb < num_cblocks; cb+=num_streams) {
      for (int d3 = 0; d3 < c_blocks; d3++) {
#pragma omp parallel for  
	for (int k=0; k < num_streams; k++) {
          int tensor_offset = (cb+k)*dims[3]; 
          const int block_offset = tensor_offset + d3*block_size; 
          cudaMemset(&d_absmax[k], 0, sizeof(T));
          cudaMemset(&d_min[k], 0, sizeof(T));
          cudaMemset(&d_max[k], 0, sizeof(T));
          cudaMemset(&d_sum[k], 0, sizeof(float));
          if (pruning_algo == 0) { 
            max_reduce<<<grid, block, block*sizeof(float), streams[k]>>>(input_flat, &d_absmax[k], block_offset, block_size); 
 	  } else {
            min_max_reduce<<<grid, block, 2*block*sizeof(float), streams[k]>>>(input_flat, &d_min[k], &d_max[k], 
				block_offset, block_size); 
            reduce_interquartile_sum<<<grid, block, block*sizeof(float), streams[k]>>>(input_flat, &d_sum[k], 
				&d_min[k], &d_max[k], block_offset, block_size); 
	    estimate_mode_range <<<grid, block, 0, streams[k]>>>(&d_absmax[k], &d_sum[k], &d_min[k], &d_max[k], block_size);
	  }
          QuantEmuCudaKernel <<<grid, block, 0, streams[k]>>>(
					unsigned_data, 
					mbits, 
					&d_absmax[k], 
					rmode, 
					block_offset, 
					block_size, 
					input_flat, 
					output_flat);
//          cudaStreamSynchronize(streams[k]);
        }
      }
    }
    //cudaDeviceSynchronize();
    for (int s =0; s < num_streams; s++ ) cudaStreamDestroy(streams[s]);
    cudaFree(d_absmax);
    cudaFree(d_min);
    cudaFree(d_max);
    cudaFree(d_sum);
  }
};
template struct BlockC_QuantEmuFunctor<GPUDevice, float>;
template struct BlockC_QuantEmuFunctor<GPUDevice, Eigen::half>;

template <typename T>
struct BlockCHW_QuantEmuFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int pruning_algo, int unsigned_data, int mbits, int *dims, 
	int cblock_size, int rmode, T *in, T *out) {

    int chw_blocks =  dims[1]/cblock_size; 
    if ((dims[1]%cblock_size) || (dims[1] < cblock_size)) { chw_blocks = 1; cblock_size = dims[1];}
    int block_size = cblock_size*dims[2]*dims[3];

    T *input_flat = in; 
    T *output_flat = out; 
    int block = CUBLOCK_SIZE; //(CUBLOCK_SIZE > block_size)?block_size:CUBLOCK_SIZE; 
    int grid = (block_size + (block -1))/block;  

    const int num_streams = dims[0];  
    cudaStream_t streams[num_streams];
    T *d_absmax;
    T *d_min;
    T *d_max;
    float *d_sum;
    cudaMalloc((void**) &d_absmax, num_streams*sizeof(T));
    cudaMalloc((void**) &d_min, num_streams*sizeof(T));
    cudaMalloc((void**) &d_max, num_streams*sizeof(T));
    cudaMalloc((void**) &d_sum, num_streams*sizeof(float));

    for (int s =0; s < num_streams; s++ ) cudaStreamCreate(&streams[s]);

    for (int d0 = 0; d0 < dims[0]; d0+=num_streams) {
      for (int d1 = 0; d1 < chw_blocks; d1++) {
#pragma omp parallel for 
        for (int k=0; k < num_streams; k++) {
          int tensor_offset  = (d0+k)*dims[1]*dims[2]*dims[3];  
          int block_offset = tensor_offset + d1*block_size; 
          cudaMemset(&d_absmax[k], 0, sizeof(T));
          cudaMemset(&d_min[k], 0, sizeof(T));
          cudaMemset(&d_max[k], 0, sizeof(T));
          cudaMemset(&d_sum[k], 0, sizeof(float));

          if (pruning_algo == 0) { 
            max_reduce<<<grid, block, block*sizeof(float), streams[k]>>>(input_flat, &d_absmax[k], block_offset, block_size); 
 	  } else {
            min_max_reduce<<<grid, block, 2*block*sizeof(float), streams[k]>>>(input_flat, &d_min[k], &d_max[k], 
					block_offset, block_size); 
            reduce_interquartile_sum<<<grid, block, block*sizeof(float), streams[k]>>>(input_flat, 
					&d_sum[k], &d_min[k], &d_max[k], block_offset, block_size); 
	    estimate_mode_range <<<grid, block, 0, streams[k]>>>(&d_absmax[k], &d_sum[k], &d_min[k], &d_max[k], block_size);
   	  }
          QuantEmuCudaKernel <<<grid, block, 0, streams[k]>>>(
					unsigned_data, 
					mbits, 
					&d_absmax[k], 
					rmode, 
					block_offset, 
					block_size, 
					input_flat, 
					output_flat);
//          cudaStreamSynchronize(streams[k]);
        }
      }
//      cudaDeviceSynchronize();
    }
    for (int s =0; s < num_streams; s++ ) cudaStreamDestroy(streams[s]);
    cudaFree(d_absmax);
    cudaFree(d_min);
    cudaFree(d_max);
    cudaFree(d_sum);
  }
};
template struct BlockCHW_QuantEmuFunctor<GPUDevice, float>;
template struct BlockCHW_QuantEmuFunctor<GPUDevice, Eigen::half>;

template <typename T>
struct LowpFloatQuantEmuFunctor <GPUDevice, T> {
  void operator()(const GPUDevice& d, int mbits, int exp_bits, int rmode, int size, const T* in, T* out) {
    //std::cout << " Inside the LowpFloatQuantEmuFunctor GPU version "<< size  << std::endl; 
    int block = CUBLOCK_SIZE; 
    int grid = ( size + (CUBLOCK_SIZE-1))/CUBLOCK_SIZE; 
    QuantEmuLowpCudaKernel <<<grid, block, 0, d.stream()>>>(mbits, exp_bits, rmode, size, in, out); 
    //cudaStreamSynchronize(d.stream());
    //cudaDeviceSynchronize();
  }
};
template struct LowpFloatQuantEmuFunctor<GPUDevice, float>;
template struct LowpFloatQuantEmuFunctor<GPUDevice, Eigen::half>;

template <typename T>
struct PositQuantEmuFunctor <GPUDevice, T> {
  void operator()(const GPUDevice& d, int m_bits, int es_bits, int size, const T* in, T* out) {
    //std::cout << " Inside the PositQuantEmuFunctor GPU version "<< size  << std::endl; 
    int block = CUBLOCK_SIZE; 
    int grid = ( size + (CUBLOCK_SIZE-1))/CUBLOCK_SIZE; 
    QuantEmuPositCudaKernel<<<grid, block, 0, d.stream()>>>(m_bits, es_bits, size, in, out); 
    //cudaStreamSynchronize(d.stream());
    //cudaDeviceSynchronize();
  }
};
template struct PositQuantEmuFunctor<GPUDevice, float>;
template struct PositQuantEmuFunctor<GPUDevice, Eigen::half>;

#endif  // GOOGLE_CUDA
