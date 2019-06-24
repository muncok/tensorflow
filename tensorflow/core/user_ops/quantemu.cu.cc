#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "quantemu.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <limits>
#include "posit_types.h" 
#include <curand_kernel.h> 

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

#define CUBLOCK_SIZE 512 

typedef union half_t { 
   unsigned short u; 
   __half f; 
   struct { 
        unsigned int mantissa : 10; 
	unsigned int exponent : 5;
	unsigned int sign : 1;
   } parts;
} __half_t; 

__device__ 
static inline uint32_t rotl_(const uint32_t x, int k) {
	return (x << k) | (x >> (32 - k));
}

__device__
static uint32_t  s[4] = {0x76B5DBC3, 0x532CB7BF, 0x6AFA41C3, 0x28DBD9F7};

__device__ 
uint32_t _xorshf_rand(void) {
	const uint32_t result_plus = s[0] + s[3];
	const uint32_t t = s[1] << 9;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;

	s[3] = rotl_(s[3], 11);

	return result_plus;
}
#if 10
__device__
static uint32_t  s1[4] = {1387366120, 279844183, 888998500, 1099633400}; 
__device__
static uint32_t  s2[4] = {2034269327, 2125325156, 1209715489, 193165672};
__device__
static uint32_t  s3[4] = {1555452618, 650181557, 883695203, 62767784};
__device__
static uint32_t  s4[4] = {419524804, 2146478152, 480059239, 1468956197};
__device__
static uint32_t  s5[4] = {1252084877, 500390994, 977516591, 1950666000}; 
__device__
static uint32_t  s6[4] = {393659750, 834151069, 1477014702, 734008143};
__device__
static uint32_t  s7[4] = {1983400973, 116410309, 2110188261, 2019272068}; 
__device__
static uint32_t  s8[4] = {187709636, 28336299, 419632041, 1774181187}; 
__device__
static uint32_t  s9[4] = {702309618, 407781555, 1512057936, 1868769368}; 
__device__
static uint32_t  s10[4] = {510001215, 966559856, 776583255, 147562106};
__device__
static uint32_t  s11[4] = {127180605, 1881312534, 478635452, 814821902}; 
__device__
static uint32_t  s12[4] = {733990058, 1889991804, 1108257970, 1093480892}; 
__device__
static uint32_t  s13[4] = {427374380, 416747337, 558000409, 1594848927}; 
__device__
static uint32_t  s14[4] = {444870959, 1595722866, 1064124488, 363710254}; 
__device__
static uint32_t  s15[4] = {703721499, 389640783, 1002360059, 1427395742}; 
__device__
static uint32_t  s16[4] = {1295231497, 1254972431, 1423497865, 861918264};

__device__
static uint32_t  *sptr[16] = {s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16};
__device__ 
uint32_t _xorshf_rand_with_seed(uint32_t *ps) {
	const uint32_t result_plus = ps[0] + ps[3];
	const uint32_t t = ps[1] << 9;

	ps[2] ^= ps[0];
	ps[3] ^= ps[1];
	ps[1] ^= ps[2];
	ps[0] ^= ps[3];

	ps[2] ^= t;

	ps[3] = rotl_(ps[3], 11);

	return result_plus;
}
#endif 
#include "posit_impl_cuda.h" 

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
    shared_min[tid] = std::numeric_limits<float>::max(); 
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
    shared_min[tid] = std::numeric_limits<float>::max(); 
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
  sfquant = (float)(quant_max / __half2float(*absmax));
  sfdequant = (float)1.0/sfquant;

  int tid = threadIdx.x;
  int gid = (blockDim.x * blockIdx.x) + tid + block_offset;
  int size = block_offset + block_size; 
  float fabsfmax = __half2float(*absmax); 
 
  for (gid; gid < size; gid += blockDim.x*gridDim.x) {
      float inval = __half2float(in[gid]);
      /* saturate anything larger than fmax_val to fmax_val */
      inval = fminf (inval, fabsfmax); 
      inval = fmaxf (inval, -fabsfmax);
      /* write back saturated value */ 
      // in[gid] = __float2half_rn(inval);  /* no need for write back */

      /* round to nearest even */ 
      int ival = __half2int_rn (inval * sfquant);
      out[gid] = __float2half_rn (ival * sfdequant);
  }
}

__global__ 
void QuantEmuCudaKernel(
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
  sfquant = (float)(quant_max / *absmax);
  sfdequant = (float)1.0/sfquant;

  int tid = threadIdx.x;
  int gid = (blockDim.x * blockIdx.x) + tid + block_offset;
  int size = block_offset + block_size; 
  float fabsfmax = *absmax; 
 
  for (gid; gid < size; gid += blockDim.x*gridDim.x) {
      float inval = in[gid];
      /* saturate anything larger than fmax_val to fmax_val */
      inval = fminf (inval, fabsfmax); 
      inval = fmaxf (inval, -fabsfmax);
      /* write back saturated value */ 
      // in[gid] = inval; /* no need for write back */ 
      /* round to the nearest even */ 
      int ival = __float2int_rn (inval * sfquant);
      out[gid] = ival * sfdequant;
  }
}

__global__ 
void QuantEmuCudaKernel_Unsigned(
	int mbits, 
	Eigen::half *max, 
	Eigen::half *min, 
	int rmode, 
	const int block_offset, 
	const int block_size, 
	Eigen::half *in, 
	Eigen::half *out) 
{
  int quant_max, quant_min; 
  float sfquant;
  float sfdequant;

  quant_max = pow(2, mbits) - 1;
  quant_min = 0; 
  float max_float = __half2float(*max); 
  float min_float = __half2float(*min); 

  sfquant = (float)((quant_max - quant_min) / (max_float - min_float));
  sfdequant = (float)1.0/sfquant;

  int tid = threadIdx.x;
  int gid = (blockDim.x * blockIdx.x) + tid + block_offset;
  int size = block_offset + block_size; 
 
  for (gid; gid < size; gid += blockDim.x*gridDim.x) {
      float inval = __half2float(in[gid]);
      /* saturate anything larger than fmax_val to fmax_val */
      inval = fminf (inval, max_float); 
      inval = fmaxf (inval, min_float);
      /* write back saturated value */ 
      // in[gid] = __float2half_rn(inval);  /* no need for write back */

      /* shift to positive domain */ 
      inval -= min_float; 
      /* round to nearest even */ 
      int ival = __half2int_rn (inval * sfquant);
      /* dequantize and shift to original regime */ 
      out[gid] = __float2half_rn ((ival * sfdequant) + min_float);
  }
}

__global__ 
void QuantEmuCudaKernel_Unsigned(
	int mbits, 
	float *max, 
	float *min, 
	int rmode, 
	const int block_offset, 
	const int block_size, 
	float* in, float* out) 
{
  int quant_max, quant_min; 
  float sfquant;
  float sfdequant;

  quant_max = pow(2, mbits) - 1;
  quant_min = 0;
  sfquant = (float)((quant_max-quant_min) / (*max-*min));
  sfdequant = (float)1.0/sfquant;

  int tid = threadIdx.x;
  int gid = (blockDim.x * blockIdx.x) + tid + block_offset;
  int size = block_offset + block_size; 
 
  for (gid; gid < size; gid += blockDim.x*gridDim.x) {
      float inval = in[gid];
      /* saturate anything larger than fmax_val to fmax_val */
      inval = fminf (inval, *max); 
      inval = fmaxf (inval, *min);
      /* write back saturated value */ 
      // in[gid] = inval; /* no need for write back */
      /* shift to positive domain */ 
      inval -= *min;
      /* round to the nearest even */ 
      int ival = __float2int_rn (inval * sfquant);
      /* dequantize and shift to normal regime */ 
      out[gid] = (ival * sfdequant) + *min;
  }
}


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
  int lshift = 10 - (mbits - non_mant_bits);
  int rshift = lshift - 3; /* shift to preserve rounding bits */ 
  int exp_max = (0x1 << exp_bits-1) - 1; 
  int exp_min = 1 - exp_max; 

  unsigned short rne_mask = 0; /* round to nearest even mask */ 
  unsigned short sr_mask = 0;  /* stochastic rounding mask */ 
  if (rmode == ROUND_RNE) rne_mask = 1;  
  if (rmode == ROUND_STOCHASTIC) sr_mask = 1;  

  unsigned short mask_mant = (unsigned short)(0xFFFF << lshift);
  unsigned short mask_mant_grs = (unsigned short)(0xFFFF << rshift);
   /* mask to extract G(gaurd), R (round), S (sticky) bits */ 
  unsigned short lsbGRS = 0xF << rshift; 

//  curandStateXORWOW_t rand_state; 
//  curand_init (0, 0x28DBD9F7, 0, &rand_state);

  for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size; gid += blockDim.x * gridDim.x) {
      __half_t h; 
      float inval = in[gid];
      __half  hval = __float2half_rn(inval); 
      h.f = hval;
     
      unsigned short mant_grs = (h.u & mask_mant_grs); 
      //unsigned short not_denorm = (((h.u & 0x7FFF) >> 10) > 0); 
      unsigned short not_denorm = ((((h.u & 0x7FFF) >> 10) & 0x1F) > 0); 
      unsigned short is_denorm = (not_denorm == 0)?1:0;
      unsigned short must_round = not_denorm ? not_denorm : (((h.u >> 8)& 0x3) < 0x3); 

#if 10
      /* stochastic with 16 seeds */ 
      int seed_index = (gid/16); 
      unsigned short rand = (unsigned short) _xorshf_rand_with_seed(sptr[(seed_index%16)]);
#endif 
      /* stochastic rounding */ 
//      unsigned short rand = (unsigned short) _xorshf_rand();
      /* apply stochastic rounding before truncation if sr_mask is enabled */ 
      //h.u += must_round * not_denorm * sr_mask * (rand & 0xFF); 
      h.u += not_denorm * sr_mask * (rand & 0xFF); 

      /* round to nearest even after truncation if rne_mask is enabled */ 
      unsigned short rmask_tie = ((mant_grs & lsbGRS) >> rshift);  
      unsigned short rmask = (rmask_tie & 0x7);  
      //h.u += must_round * rne_mask * (((rmask > 0x4) || (rmask_tie == 0xC) ) << lshift); 
      h.u += rne_mask * (((rmask > 0x4) || (rmask_tie == 0xC) ) << lshift); 

      /* stochastic round denormals --> nearest rounding */ 
      //h.u += must_round * is_denorm * sr_mask * (((rmask > 0x4) || (rmask_tie == 0xC) ) << lshift); 
      h.u += is_denorm * sr_mask * (((rmask > 0x4) || (rmask_tie == 0xC) ) << lshift); 

       /* truncation */ 
      h.u = (h.u & mask_mant); 
 
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
  int lshift = 10 - (mbits - non_mant_bits);
  int rshift = lshift - 3; /* shift to preserve rounding bits */ 
  int exp_max = (0x1 << exp_bits-1) - 1; 
  int exp_min = 1 - exp_max; 

  unsigned short rne_mask = 0; /* round to nearest even mask */ 
  unsigned short sr_mask = 0;  /* stochastic rounding mask */ 
  if (rmode == ROUND_RNE) rne_mask = 1;  
  if (rmode == ROUND_STOCHASTIC) sr_mask = 1;  

  unsigned short mask_mant = (unsigned short)(0xFFFF << lshift);
  unsigned short mask_mant_grs = (unsigned short)(0xFFFF << rshift);
   /* mask to extract G(gaurd), R (round), S (sticky) bits */ 
  unsigned short lsbGRS = 0xF << rshift; 

//  curandStateXORWOW_t rand_state; 
//  curand_init (0, 0x28DBD9F7, 0, &rand_state);

  for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size; gid += blockDim.x * gridDim.x) {
      __half_t h; 
      Eigen::half inval = in[gid];
      __half  hval = inval; 
      h.f = hval;

      unsigned short mant_grs = (h.u & mask_mant_grs); 
      unsigned short not_denorm = ((((h.u & 0x7FFF) >> 10) & 0x1F) > 0); 
      unsigned short is_denorm = (not_denorm == 0)?1:0;
      unsigned short must_round = not_denorm ? not_denorm : (((h.u >> 8)& 0x3) < 0x3); 
#if 10
      /* stochastic with 16 seeds */ 
      int seed_index = (gid/16); 
      unsigned short rand = (unsigned short) _xorshf_rand_with_seed(sptr[(seed_index%16)]);
#else 
      /* stochastic rounding */ 
      unsigned short rand = (unsigned short) _xorshf_rand();
#endif 
      //unsigned short rand = (unsigned short) curand(&rand_state);
      /* apply stochastic rounding before truncation if sr_mask is enabled */ 
      //h.u += sr_mask * (rand & 0x00FF); 
      //h.u += must_round * not_denorm * sr_mask * (rand & 0xFF); 
      h.u += not_denorm * sr_mask * (rand & 0xFF); 

      /* round to nearest even after truncation if rne_mask is enabled */ 
      unsigned short rmask_tie = ((mant_grs & lsbGRS) >> rshift);  
      unsigned short rmask = (rmask_tie & 0x7);  
      //h.u += must_round * rne_mask * (((rmask > 0x4) || (rmask_tie == 0xC) ) << lshift); 
      h.u += rne_mask * (((rmask > 0x4) || (rmask_tie == 0xC) ) << lshift); 
      /* stochastic round denormals --> nearest rounding */ 
      //h.u += must_round * is_denorm * sr_mask * (((rmask > 0x4) || (rmask_tie == 0xC) ) << lshift); 
      h.u += is_denorm * sr_mask * (((rmask > 0x4) || (rmask_tie == 0xC) ) << lshift); 

      /* truncation */ 
      h.u = (h.u & mask_mant); 

      Eigen::half outval = h.f; 
      out[gid] = outval;
  }
}


__global__ 
void QuantEmuLOG2CudaKernel(
	const int size, 
	const float *in, 
	float *out) 
{
  unsigned int  mask_exp = (unsigned int)(0x7F800000);
  int exp7b_max = 63; 
  int exp7b_min = -62; 

  for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size; gid += blockDim.x * gridDim.x) {
   
      UFLOAT32 uf; 
      uf.f = in[gid];
      int log2_7b = (((uf.u & mask_exp) >> 23) - 127); 
      log2_7b = min(exp7b_max, log2_7b); 
      log2_7b = max(exp7b_min, log2_7b); 
      log2_7b += 127; 
      unsigned int ival = (uf.u & 0x80000000) | (log2_7b << 23); 
      uf.u = ival; 
      out[gid] = uf.f;
  }
}

__global__ 
void QuantEmuLOG2CudaKernel(
	const int size, 
	const Eigen::half* in, 
	Eigen::half* out) 
{
  unsigned short mask_exp = (unsigned short)(0x7C00);
  short exp4b_max = 7; 
  short exp4b_min = -6; 

  for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size; gid += blockDim.x * gridDim.x) {
   
      __half_t h; 
      Eigen::half inval = in[gid];
      __half  hval = inval; 
      h.f = hval;

      short log2_4b = (((h.u & mask_exp) >> 10) - 15); 
      log2_4b = min(exp4b_max, log2_4b); 
      log2_4b = max(exp4b_min, log2_4b); 
      log2_4b += 15; 
      unsigned short ival = (h.u & 0x8000) | (log2_4b << 10); 
      h.u = ival;
      Eigen::half outval = h.f; 
      out[gid] = outval;
  }
}

__global__ 
void QuantEmuPositCudaKernel(
	int m_bits, 
	int es_bits, 
	int rmode, 
	const int size, 
	const float *in, 
	float *out) 
{
  for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size; gid += blockDim.x * gridDim.x) {
      float inval = in[gid];
      POSIT_UTYPE pval =_pack_posit(_unpack_float(inval), m_bits, es_bits, rmode); 
      float outval = _pack_float (_unpack_posit(pval, m_bits, es_bits));  
      out[gid] = outval;
  }
}
__global__ 
void QuantEmuPositCudaKernel(
	int m_bits, 
	int es_bits, 
	int rmode, 
	const int size, 
	const Eigen::half *in, 
	Eigen::half *out) 
{
  for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size; gid += blockDim.x * gridDim.x) {
      __half inval = in[gid];
      POSIT_UTYPE pval =_pack_posit(_unpack_float(__half2float(inval)), m_bits, es_bits, rmode); 
      float outval = _pack_float (_unpack_posit(pval, m_bits, es_bits));  
      out[gid] = __float2half_rn(outval);
  }
}

__global__ 
void QuantEmuBfloat16CudaKernel(
	int rmode, 
	const int size, 
	const float *in, 
	float *out) 
{
  int lshift = 16;
  int rshift = lshift - 3; /* shift to preserve rounding bits */ 
  unsigned int mask_mant = (unsigned int)(0xFFFFFFFF << lshift);
  unsigned int mask_mant_grs = (unsigned int)(0xFFFFFFFF << rshift);
   /* mask to extract G(gaurd), R (round), S (sticky) bits */ 
  unsigned int lsbGRS = 0xF << rshift; 

  unsigned short rne_mask = 0; /* round to nearest even mask */ 
  unsigned short sr_mask = 0;  /* stochastic rounding mask */ 
  if (rmode == ROUND_RNE) rne_mask = 1;  
  if (rmode == ROUND_STOCHASTIC) sr_mask = 1;  

  for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size; gid += blockDim.x * gridDim.x) {
      UFLOAT32 uf; 
      float inval = in[gid];
      uf.f = inval; 
      unsigned int mant_grs = (uf.u & mask_mant_grs); 

      /* stochastic with 16 seeds */ 
      int seed_index = (gid/16); 
      unsigned short rand = (unsigned short) _xorshf_rand_with_seed(sptr[(seed_index%16)]);
      /* stochastic rounding with 16-bit random number */ 
      uf.u += sr_mask * rand; 
    
      /* truncation */ 
      uf.u &= mask_mant; 

      /* round to nearest even after truncation if rne_mask is enabled */ 
      unsigned int rmask_tie = ((mant_grs & lsbGRS) >> rshift);  
      unsigned int rmask = (rmask_tie & 0x7);  
      uf.u += rne_mask * (((rmask > 0x4) || (rmask_tie == 0xC) ) << lshift); 

      out[gid] = uf.f;
  }
}
__global__ 
void QuantEmuBfloat16CudaKernel(
	int rmode, 
	const int size, 
	const Eigen::half* in, 
	Eigen::half* out) 
{
  int lshift = 16;
  int rshift = lshift - 3; /* shift to preserve rounding bits */ 
  unsigned int mask_mant = (unsigned int )(0xFFFFFFFF << lshift);
  unsigned int mask_mant_grs = (unsigned int)(0xFFFFFFFF << rshift);
   /* mask to extract G(gaurd), R (round), S (sticky) bits */ 
  unsigned int lsbGRS = 0xF << rshift; 

  unsigned short rne_mask = 0; /* round to nearest even mask */ 
  unsigned short sr_mask = 0;  /* stochastic rounding mask */ 
  if (rmode == ROUND_RNE) rne_mask = 1;  
  if (rmode == ROUND_STOCHASTIC) sr_mask = 1;  

  for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size; gid += blockDim.x * gridDim.x) {
      UFLOAT32 uf; 

      float inval = __half2float(in[gid]);
      uf.f = inval; 
      unsigned int mant_grs = (uf.u & mask_mant_grs); 

      /* stochastic with 16 seeds */ 
      int seed_index = (gid/16); 
      unsigned short rand = (unsigned short) _xorshf_rand_with_seed(sptr[(seed_index%16)]);
      /* stochastic rounding with 16-bit random number */ 
      uf.u += sr_mask * rand; 

      /* truncation */ 
      uf.u &= mask_mant; 

      /* round to nearest even after truncation if rne_mask is enabled */ 
      unsigned int rmask_tie = ((mant_grs & lsbGRS) >> rshift);  
      unsigned int rmask = (rmask_tie & 0x7);  
      uf.u += rne_mask * (((rmask > 0x4) || (rmask_tie == 0xC) ) << lshift); 

      out[gid] = __float2half_rn(uf.f);
  }
}
/* 
 * Emulate FP16 without denormals. 
 */
__global__ 
void QuantEmuModFP16CudaKernel(
	const int size, 
	const float *in, 
	float *out) 
{
  for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size; gid += blockDim.x * gridDim.x) {
      __half_t h; 
      float inval = in[gid];
      __half  hval = __float2half_rn(inval); 
      h.f = hval;
      unsigned short not_denorm = ((((h.u & 0x7FFF) >> 10) & 0x1F) > 0); 
      unsigned short is_denorm = (not_denorm == 0)?1:0;
      h.u *= !is_denorm; 
      float outval = __half2float(h.f);
      out[gid] = outval;
  }
}
__global__ 
void QuantEmuModFP16CudaKernel(
	const int size, 
	const Eigen::half* in, 
	Eigen::half* out) 
{
  for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size; gid += blockDim.x * gridDim.x) {
      __half_t h; 
      Eigen::half inval = in[gid];
      __half  hval = inval; 
      h.f = hval;
      unsigned short not_denorm = ((((h.u & 0x7FFF) >> 10) & 0x1F) > 0); 
      unsigned short is_denorm = (not_denorm == 0)?1:0;
      h.u *= !is_denorm;
      Eigen::half outval = h.f; 
      out[gid] = outval;
  }
}

/* Define the GPU implementation that launches the CUDA kernel. */
template <typename T>
struct QuantEmuFunctor<GPUDevice, T> { 
  void operator()(const GPUDevice& d, int unsigned_data, int mbits, int rmode, int size, T* in, T* out) {
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
    if(unsigned_data) {
      min_max_reduce<<<grid, block, 2*block*sizeof(float), d.stream()>>>(in, d_min, d_max, 0, size); 
      QuantEmuCudaKernel_Unsigned <<<grid, block, 0, d.stream()>>>(mbits, d_max, d_min, rmode, 0, size, in, out);
    } else { 
      max_reduce<<<grid, block, block*sizeof(float), d.stream()>>>(in, d_absmax, 0, size); 
      QuantEmuCudaKernel <<<grid, block, 0, d.stream()>>>(mbits, d_absmax, rmode, 0, size, in, out);
    }
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
  void operator()(const GPUDevice& d, int unsigned_data, int mbits, int *dims , 
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
//#pragma omp parallel for  
	for (int k=0; k < num_streams; k++) {
          int tensor_offset = (cb+k)*dims[3]; 
          const int block_offset = tensor_offset + d3*block_size; 
          cudaMemset(&d_absmax[k], 0, sizeof(T));
          cudaMemset(&d_min[k], 0, sizeof(T));
          cudaMemset(&d_max[k], 0, sizeof(T));
          cudaMemset(&d_sum[k], 0, sizeof(float));
          if (unsigned_data) {
            min_max_reduce<<<grid, block, 2*block*sizeof(float), streams[k]>>>(input_flat, &d_min[k], &d_max[k], 
				block_offset, block_size); 
            QuantEmuCudaKernel_Unsigned <<<grid, block, 0, streams[k]>>>(
					mbits, 
					&d_max[k], 
					&d_min[k], 
					rmode, 
					block_offset, 
					block_size, 
					input_flat, 
					output_flat);
 	  } else {
            max_reduce<<<grid, block, block*sizeof(float), streams[k]>>>(input_flat, &d_absmax[k], block_offset, block_size); 
            QuantEmuCudaKernel <<<grid, block, 0, streams[k]>>>(
					mbits, 
					&d_absmax[k], 
					rmode, 
					block_offset, 
					block_size, 
					input_flat, 
					output_flat);
          }
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
  void operator()(const GPUDevice& d, int unsigned_data, int mbits, int *dims, 
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
//#pragma omp parallel for 
        for (int k=0; k < num_streams; k++) {
          int tensor_offset  = (d0+k)*dims[1]*dims[2]*dims[3];  
          int block_offset = tensor_offset + d1*block_size; 
          cudaMemset(&d_absmax[k], 0, sizeof(T));
          cudaMemset(&d_min[k], 0, sizeof(T));
          cudaMemset(&d_max[k], 0, sizeof(T));
          cudaMemset(&d_sum[k], 0, sizeof(float));

          if (unsigned_data) { 
            min_max_reduce<<<grid, block, 2*block*sizeof(float), streams[k]>>>(input_flat, &d_min[k], &d_max[k], 
					block_offset, block_size); 
            QuantEmuCudaKernel_Unsigned <<<grid, block, 0, streams[k]>>>(
					mbits, 
					&d_max[k], 
					&d_min[k], 
					rmode, 
					block_offset, 
					block_size, 
					input_flat, 
					output_flat);
 	  } else {
            max_reduce<<<grid, block, block*sizeof(float), streams[k]>>>(input_flat, &d_absmax[k], block_offset, block_size); 
            QuantEmuCudaKernel <<<grid, block, 0, streams[k]>>>(
					mbits, 
					&d_absmax[k], 
					rmode, 
					block_offset, 
					block_size, 
					input_flat, 
					output_flat);
	  }
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
struct Log2QuantEmuFunctor <GPUDevice, T> {
  void operator()(const GPUDevice& d, int size, const T* in, T* out) {
    //std::cout << " Inside the Log2QuantEmuFunctor GPU version "<< size  << std::endl; 
    int block = CUBLOCK_SIZE; 
    int grid = ( size + (CUBLOCK_SIZE-1))/CUBLOCK_SIZE; 
    QuantEmuLOG2CudaKernel <<<grid, block, 0, d.stream()>>>(size, in, out); 
    //cudaStreamSynchronize(d.stream());
    //cudaDeviceSynchronize();
  }
};
template struct Log2QuantEmuFunctor<GPUDevice, float>;
template struct Log2QuantEmuFunctor<GPUDevice, Eigen::half>;

template <typename T>
struct PositQuantEmuFunctor <GPUDevice, T> {
  void operator()(const GPUDevice& d, int m_bits, int es_bits, int rmode, int size, const T* in, T* out) {
    //std::cout << " Inside the PositQuantEmuFunctor GPU version "<< size  << std::endl; 
    int block = CUBLOCK_SIZE; 
    int grid = ( size + (CUBLOCK_SIZE-1))/CUBLOCK_SIZE; 
    QuantEmuPositCudaKernel<<<grid, block, 0, d.stream()>>>(m_bits, es_bits, rmode, size, in, out); 
    //cudaStreamSynchronize(d.stream());
    //cudaDeviceSynchronize();
  }
};
template struct PositQuantEmuFunctor<GPUDevice, float>;
template struct PositQuantEmuFunctor<GPUDevice, Eigen::half>;

template <typename T>
struct BfloatQuantEmuFunctor <GPUDevice, T> {
  void operator()(const GPUDevice& d, int rmode, int size, const T* in, T* out) {
    //std::cout << " Inside the LowpFloatQuantEmuFunctor GPU version "<< size  << std::endl; 
    int block = CUBLOCK_SIZE; 
    int grid = ( size + (CUBLOCK_SIZE-1))/CUBLOCK_SIZE; 
    QuantEmuBfloat16CudaKernel <<<grid, block, 0, d.stream()>>>(rmode, size, in, out); 
    //cudaStreamSynchronize(d.stream());
    //cudaDeviceSynchronize();
  }
};

template struct BfloatQuantEmuFunctor<GPUDevice, float>;
template struct BfloatQuantEmuFunctor<GPUDevice, Eigen::half>;

template <typename T>
struct ModFP16QuantEmuFunctor <GPUDevice, T> {
  void operator()(const GPUDevice& d, int size, const T* in, T* out) {
    //std::cout << " Inside the LowpFloatQuantEmuFunctor GPU version "<< size  << std::endl; 
    int block = CUBLOCK_SIZE; 
    int grid = ( size + (CUBLOCK_SIZE-1))/CUBLOCK_SIZE; 
    QuantEmuModFP16CudaKernel<<<grid, block, 0, d.stream()>>>(size, in, out); 
    //cudaStreamSynchronize(d.stream());
    //cudaDeviceSynchronize();
  }
};

template struct ModFP16QuantEmuFunctor<GPUDevice, float>;
template struct ModFP16QuantEmuFunctor<GPUDevice, Eigen::half>;

#endif  // GOOGLE_CUDA
