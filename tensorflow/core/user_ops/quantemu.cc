#include "quantemu.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <assert.h>

using namespace tensorflow;  

#include <immintrin.h> 
#include <omp.h> 
  
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("QuantizeEmu")
    //.Attr("T: {float, Eigen::half, int32} = DT_FLOAT")
    .Attr("T: type")
    .Input("to_quantize: T")
    .Output("quantized: T")
    .Attr("data_format: {'unknown', 'channels_first', 'channels_last'}")
    .Attr("allocate_copy: int = 0") 
    .Attr("data_type: int = 0")
    .Attr("precision: int = 23")
    .Attr("exponent_bits: int = 5")
    .Attr("channel_blocking_type: int = 0")
    .Attr("channels_per_block: int = 0")
    .Attr("round_mode: int = 0")
    .Attr("log_tensor: int = 0")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK(); 
    });

#define ROUNDING_CTRL (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)
#ifdef AVX512F
__m512 _mm512_cvtpsph_ps(__m512 reg)
{
  __m256i hreg  = _mm512_cvtps_ph(reg, ROUNDING_CTRL);
  return  _mm512_cvtph_ps (hreg);
}
#elif AVX 
__m256 _mm256_cvtpsph_ps(__m256 reg)
{
  __m128i hreg  = _mm256_cvtps_ph(reg, ROUNDING_CTRL);
  return  _mm256_cvtph_ps (hreg);
}
#endif 

unsigned short __float2half_rn (float inval) 
{
  unsigned short out[8] = {0};
  __m128i vout = _mm_cvtps_ph (_mm_set1_ps (inval),(_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
  _mm_store_si128 ((__m128i*)&out[0], vout); 
  return out[0]; 
}

float __half2float(unsigned short hval) 
{
  float out[8] = {0}; 
  __m128 vout = _mm_cvtph_ps(_mm_set1_epi16(hval)); 
  _mm_store_ps (&out[0], vout); 
  return out[0]; 
}

static inline uint32_t rotl(const uint32_t x, int k) {
	return (x << k) | (x >> (32 - k));
}


static uint32_t  s_[4] = {0x76B5DBC3, 0x532CB7BF, 0x6AFA41C3, 0x28DBD9F7};

uint32_t xorshf_rand(void) {
	const uint32_t result_plus = s_[0] + s_[3];
	const uint32_t t = s_[1] << 9;

	s_[2] ^= s_[0];
	s_[3] ^= s_[1];
	s_[1] ^= s_[2];
	s_[0] ^= s_[3];

	s_[2] ^= t;

	s_[3] = rotl(s_[3], 11);

	return result_plus;
}
static uint32_t  s1_[4] = {1387366120, 279844183, 888998500, 1099633400}; 
static uint32_t  s2_[4] = {2034269327, 2125325156, 1209715489, 193165672};
static uint32_t  s3_[4] = {1555452618, 650181557, 883695203, 62767784};
static uint32_t  s4_[4] = {419524804, 2146478152, 480059239, 1468956197};
static uint32_t  s5_[4] = {1252084877, 500390994, 977516591, 1950666000}; 
static uint32_t  s6_[4] = {393659750, 834151069, 1477014702, 734008143};
static uint32_t  s7_[4] = {1983400973, 116410309, 2110188261, 2019272068}; 
static uint32_t  s8_[4] = {187709636, 28336299, 419632041, 1774181187}; 
static uint32_t  s9_[4] = {702309618, 407781555, 1512057936, 1868769368}; 
static uint32_t  s10_[4] = {510001215, 966559856, 776583255, 147562106};
static uint32_t  s11_[4] = {127180605, 1881312534, 478635452, 814821902}; 
static uint32_t  s12_[4] = {733990058, 1889991804, 1108257970, 1093480892}; 
static uint32_t  s13_[4] = {427374380, 416747337, 558000409, 1594848927}; 
static uint32_t  s14_[4] = {444870959, 1595722866, 1064124488, 363710254}; 
static uint32_t  s15_[4] = {703721499, 389640783, 1002360059, 1427395742}; 
static uint32_t  s16_[4] = {1295231497, 1254972431, 1423497865, 861918264};

static uint32_t  *sptr_[16] = {s1_, s2_, s3_, s4_, s5_, s6_, s7_, s8_, s9_, s10_, s11_, s12_, s13_, s14_, s15_, s16_};
uint32_t xorshf_rand_with_seed(uint32_t *ps) {
	const uint32_t result_plus = ps[0] + ps[3];
	const uint32_t t = ps[1] << 9;

	ps[2] ^= ps[0];
	ps[3] ^= ps[1];
	ps[1] ^= ps[2];
	ps[0] ^= ps[3];

	ps[2] ^= t;

	ps[3] = rotl(ps[3], 11);

	return result_plus;
}

/* Posit implementation */ 
#include "posit_impl.h" 

void QuantEmuIntCPUKernel(
	int mbits, 
	int rmode, 
	const int size, 
	const float *in, 
	float *out) 
{
    float absmax = (float)0.0; 
    #pragma omp parallel for reduction(max: absmax)
    for (int i=0; i < size; i++) if (fabs(in[i]) > absmax) absmax = fabs(in[i]);  

    int quant_max = pow(2, mbits-1) - 1;
    float sfquant = (float)(quant_max/absmax);
    float sfdequant = (float)1.0/sfquant;
    float sfquant_plus8b = (float)((quant_max*256) /absmax);  /* use 8 extra bits to round */ 
    /* mask to extract G(gaurd), R (round), S (sticky) bits */ 
    int lshift = 8;
    int rshift = lshift - 3; /* RNE: only use 3 our of 8 extra bits to round */
    unsigned int lsbGRS = 0xF << rshift; 

    int rne_mask = 0; /* round to nearest even mask */ 
    int sr_mask = 0;  /* stochastic rounding mask */ 
    if (rmode == ROUND_RNE) rne_mask = 1;  
    if (rmode == ROUND_STOCHASTIC) sr_mask = 1;  

    #pragma omp parallel for 
    for (int i = 0; i < size; ++i) {
      float inval = in[i];
      /* saturate anything larger than fmax_val to fmax_val */
      inval = std::min(inval, absmax); 
      inval = std::max(inval, -absmax); 
      int ival_plus8b = (int)(inval * sfquant_plus8b); 
      /* stochastic rounding */ 
      unsigned int rand = xorshf_rand();
      ival_plus8b += sr_mask * (rand & 0xFF); 
      /* round to nearest even after truncation if rne_mask is enabled */ 
      unsigned short rmask_tie = ((ival_plus8b & lsbGRS) >> rshift);  
      unsigned short rmask = (rmask_tie & 0x7);  
      ival_plus8b += rne_mask * (((rmask > 0x4) || (rmask_tie == 0xC) ) << lshift); 
      ival_plus8b += sr_mask * (ival_plus8b & lsbGRS); 
      
      /* truncation after rounding */ 
      int ival = ival_plus8b >> lshift; 
      out[i] = ival * sfdequant;
    }
}

void QuantEmuIntCPUKernel(
	int mbits, 
	int rmode, 
	const int size, 
	const Eigen::half *h_in, 
	Eigen::half *out) 
{
    float absmax = (float)0.0; 
    unsigned short *in = (unsigned short *) h_in; 
  
    #pragma omp parallel for reduction(max: absmax)
    for (int i=0; i < size; i++) if (fabs(__half2float(in[i])) > absmax) absmax = fabs(__half2float(in[i]));  

    int quant_max = pow(2, mbits-1) - 1;
    float sfquant = (float)(quant_max/absmax);
    float sfdequant = (float)1.0/sfquant;
    float sfquant_plus8b = (float)((quant_max*256) /absmax);  /* use 8 extra bits to round */ 
    /* mask to extract G(gaurd), R (round), S (sticky) bits */ 
    int lshift = 8;
    int rshift = lshift - 3; /* RNE: only use 3 our of 8 extra bits to round */
    unsigned int lsbGRS = 0xF << rshift; 

    int rne_mask = 0; /* round to nearest even mask */ 
    int sr_mask = 0;  /* stochastic rounding mask */ 
    if (rmode == ROUND_RNE) rne_mask = 1;  
    if (rmode == ROUND_STOCHASTIC) sr_mask = 1;  

    #pragma omp parallel for 
    for (int i = 0; i < size; ++i) {
      float inval = __half2float(in[i]);
      /* saturate anything larger than fmax_val to fmax_val */
      inval = std::min(inval, absmax); 
      inval = std::max(inval, -absmax); 
      int ival_plus8b = (int)(inval * sfquant_plus8b); 
      /* stochastic rounding */ 
      unsigned int rand = xorshf_rand();
      ival_plus8b += sr_mask * (rand & 0xFF); 
      /* round to nearest even after truncation if rne_mask is enabled */ 
      unsigned short rmask_tie = ((ival_plus8b & lsbGRS) >> rshift);  
      unsigned short rmask = (rmask_tie & 0x7);  
      ival_plus8b += rne_mask * (((rmask > 0x4) || (rmask_tie == 0xC) ) << lshift); 
      ival_plus8b += sr_mask * (ival_plus8b & lsbGRS); 
      
      /* truncation after rounding */ 
      int ival = ival_plus8b >> lshift; 
      float outval = ival * sfdequant; 

      out[i] = (Eigen::half)__float2half_rn(outval);
    }
}

/* CPU specialization of actual computation. */
template <typename T>
struct QuantEmuFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int unsigned_data, int mbits, int rmode, int size, T* in, T* out) {
     QuantEmuIntCPUKernel(mbits, rmode, size, in, out);
  }
};

/* CPU specialization of actual computation. */
template <typename T>
struct BlockC_QuantEmuFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int unsigned_data, int mbits, int *dims, 
	int block_size, int rmode, T *in, T *out) {

    T *input_flat = in; 
    T *output_flat = out; 

    int c_blocks =  dims[3]/block_size; 
    if ((dims[3]%block_size) || (dims[3] < block_size)) { c_blocks = 1; block_size = dims[3];}

    for (int d0 = 0; d0 < dims[0]; d0++) {
      for (int d1 = 0; d1 < dims[1]; d1++) {
        for (int d2 = 0; d2 < dims[2]; d2++) {
          for (int d3 = 0; d3 < c_blocks; d3++) {
	    int tensor_offset = d0*dims[1]*dims[2]*dims[3] + d1*dims[2]*dims[3] + d2*dims[3]; 
            int block_offset = d3*block_size; 
  	    T *input_block = &input_flat[tensor_offset + block_offset]; 
  	    T *output_block = &output_flat[tensor_offset + block_offset]; 

            QuantEmuFunctor<CPUDevice, T>()(
                d,
                unsigned_data,
                mbits, 	
	        rmode,
                block_size,  
                input_block,
                output_block);
	  }
        }
      }
    }
  }
};

template <typename T>
struct BlockCHW_QuantEmuFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int unsigned_data, int mbits, int *dims, 
		int cblock, int rmode, T *in, T *out) {

    int chw_blocks =  dims[1]/cblock; 
    if ((dims[1]%cblock) || (dims[1] < cblock)) { chw_blocks = 1; cblock = dims[1];}
    int block_size = cblock*dims[2]*dims[3];

    T *input_flat = in; 
    T *output_flat = out; 

    for (int d0 = 0; d0 < dims[0]; d0++) {
      for (int d1 = 0; d1 < chw_blocks; d1++) {
	int tensor_offset = d0*dims[1]*dims[2]*dims[3];
        int block_offset = d1*block_size; 
  	T *input_block = &input_flat[tensor_offset + block_offset]; 
  	T *output_block = &output_flat[tensor_offset + block_offset]; 

        QuantEmuFunctor<CPUDevice, T>()(
                d,
                unsigned_data,
                mbits, 	
	        rmode,
                block_size,  
                input_block,
                output_block);
      }
    }
  }
};


void QuantEmuLowpCPUKernel(
	int mbits, 
	int exp_bits, 
	int rmode, 
	const int size, 
	const float *in, 
	float *out) 
{
    int non_mant_bits = exp_bits + 1; /* exponent + sign */
    if (mbits <= non_mant_bits) { printf("LPFP size cannot be <=5\n"); exit (0); }
    int lshift = 10 - (mbits - non_mant_bits);
    int rshift = lshift - 3; /* shift to preserve rounding bits */ 
    unsigned short mask_mant = (unsigned short)(0xFFFF << lshift);
    unsigned short mask_mant_grs = (unsigned short)(0xFFFF << rshift);
    /* mask to extract G(gaurd), R (round), S (sticky) bits */ 
    unsigned short lsbGRS = 0xF << rshift; 

    unsigned short rne_mask = 0; /* round to nearest even mask */ 
    unsigned short sr_mask = 0;  /* stochastic rounding mask */ 
    if (rmode == ROUND_RNE) rne_mask = 1;  
    if (rmode == ROUND_STOCHASTIC) sr_mask = 1;  

    #pragma omp parallel for 
    for (int i=0; i < size; i++) 
    {
      unsigned short hu = __float2half_rn(in[i]); 
     
      unsigned short mant_grs = (hu & mask_mant_grs); 
      unsigned short not_denorm = ((((hu & 0x7FFF) >> 10) & 0x1F) > 0); 
      unsigned short is_denorm = (not_denorm == 0)?1:0;

      /* stochastic rounding */ 
//      unsigned short rand = (unsigned short) xorshf_rand();
      int seed_index = (i/16); 
      unsigned short rand = (unsigned short) xorshf_rand_with_seed(sptr_[(seed_index%16)]);

      /* apply stochastic rounding before truncation if sr_mask is enabled */ 
      hu += not_denorm * sr_mask * (rand & 0xFF); 

      /* round to nearest even after truncation if rne_mask is enabled */ 
      unsigned short rmask_tie = ((mant_grs & lsbGRS) >> rshift);  
      unsigned short rmask = (rmask_tie & 0x7);  
      hu += rne_mask * (((rmask > 0x4) || (rmask_tie == 0xC) ) << lshift); 

      /* stochastic round denormals --> nearest rounding */ 
      hu += is_denorm * sr_mask * (((rmask > 0x4) || (rmask_tie == 0xC) ) << lshift); 

       /* truncation */ 
      hu = (hu & mask_mant); 
      out[i] = __half2float(hu); 
    }
}

void QuantEmuLowpCPUKernel(
	int mbits, 
	int exp_bits, 
	int rmode, 
	const int size, 
	const Eigen::half *in, 
	Eigen::half *out) 
{
    int non_mant_bits = exp_bits + 1; /* exponent + sign */
    if (mbits <= non_mant_bits) { printf("LPFP size cannot be <=5\n"); exit (0); }
    int lshift = 10 - (mbits - non_mant_bits);
    int rshift = lshift - 3; /* shift to preserve rounding bits */ 
    unsigned short mask_mant = (unsigned short)(0xFFFF << lshift);
    unsigned short mask_mant_grs = (unsigned short)(0xFFFF << rshift);
    /* mask to extract G(gaurd), R (round), S (sticky) bits */ 
    unsigned short lsbGRS = 0xF << rshift; 

    unsigned short rne_mask = 0; /* round to nearest even mask */ 
    unsigned short sr_mask = 0;  /* stochastic rounding mask */ 
    if (rmode == ROUND_RNE) rne_mask = 1;  
    if (rmode == ROUND_STOCHASTIC) sr_mask = 1;  

    #pragma omp parallel for 
    for (int i=0; i < size; i++) 
    {
      unsigned short *inptr = (unsigned short *)in; 
      unsigned short hu  = inptr[i];

      unsigned short mant_grs = (hu & mask_mant_grs); 
      unsigned short not_denorm = ((((hu & 0x7FFF) >> 10) & 0x1F) > 0); 
      unsigned short is_denorm = (not_denorm == 0)?1:0;

      /* stochastic rounding */ 
      //unsigned short rand = (unsigned short) xorshf_rand();
      int seed_index = (i/16); 
      unsigned short rand = (unsigned short) xorshf_rand_with_seed(sptr_[(seed_index%16)]);

      /* apply stochastic rounding before truncation if sr_mask is enabled */ 
      hu += not_denorm * sr_mask * (rand & 0xFF); 

      /* round to nearest even after truncation if rne_mask is enabled */ 
      unsigned short rmask_tie = ((mant_grs & lsbGRS) >> rshift);  
      unsigned short rmask = (rmask_tie & 0x7);  
      hu += rne_mask * (((rmask > 0x4) || (rmask_tie == 0xC) ) << lshift); 

      /* stochastic round denormals --> nearest rounding */ 
      hu += is_denorm * sr_mask * (((rmask > 0x4) || (rmask_tie == 0xC) ) << lshift); 

       /* truncation */ 
      hu = (hu & mask_mant); 
      out[i] = (Eigen::half)hu; 
    }
}

template <typename T>
struct LowpFloatQuantEmuFunctor <CPUDevice, T> {
  void operator()(const CPUDevice& d, int mbits, int exp_bits, int rmode, int size, const T* in, T* out) {
    //std::cout << "LowpFloatQuantEmuFunctor, non_mant_bits: " << non_mant_bits << ", mbits: " << mbits  << ", size: " << size << std::endl;
    QuantEmuLowpCPUKernel( mbits, exp_bits, rmode, size, in, out);
  }
};

void QuantEmuLOG2CPUKernel(
	const int size, 
	const float *in, 
	float *out) 
{
  unsigned int  mask_exp = (unsigned int)(0x7F800000);
  int exp7b_max = 63; 
  int exp7b_min = -62; 

  for (int i = 0; i < size; i++) {
      UFLOAT32 uf; 
      uf.f = in[i];
      int log2_7b = (((uf.u & mask_exp) >> 23) - 127); 
      log2_7b = std::min(exp7b_max, log2_7b); 
      log2_7b = std::max(exp7b_min, log2_7b); 
      log2_7b += 127; 
      unsigned int ival = (uf.u & 0x80000000) | (log2_7b << 23); 
      uf.u = ival; 
      out[i] = uf.f;
  }
}

void QuantEmuLOG2CPUKernel(
	const int size, 
	const Eigen::half* in, 
	Eigen::half* out) 
{
  unsigned short mask_exp = (unsigned short)(0x7C00);
  short exp4b_max = 7; 
  short exp4b_min = -6; 

  for (int i = 0; i < size; i++) {
      unsigned short *inptr = (unsigned short*)in; 
      unsigned short hu = inptr[i];

      short log2_4b = (((hu & mask_exp) >> 10) - 15); 
      log2_4b = std::min(exp4b_max, log2_4b); 
      log2_4b = std::max(exp4b_min, log2_4b); 
      log2_4b += 15; 
      unsigned short ival = (hu & 0x8000) | (log2_4b << 10); 
      hu = ival;
      out[i] = (Eigen::half) hu; 
  }
}

template <typename T>
struct Log2QuantEmuFunctor <CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, const T* in, T* out) {
    QuantEmuLOG2CPUKernel(size, in, out); 
  }
};

void QuantEmuPositCPUKernel(
	int m_bits, 
	int es_bits, 
	int rmode, 
	const int size, 
	const float *in, 
	float *out) 
{
  for (int i = 0; i < size; i++) {
      float inval = in[i];
      POSIT_UTYPE pval = pack_posit(unpack_float(inval), m_bits, es_bits, rmode); 
      float outval = pack_float (unpack_posit(pval, m_bits, es_bits));  
      out[i] = outval;
  }
}

void QuantEmuPositCPUKernel(
	int m_bits, 
	int es_bits, 
	int rmode, 
	const int size, 
	const Eigen::half *in, 
	Eigen::half *out) 
{
  for (int i = 0; i < size; i++) {
      unsigned short *inptr = (unsigned short *) in;
      unsigned short inval = inptr[i];
      POSIT_UTYPE pval = pack_posit(unpack_float(__half2float(inval)), m_bits, es_bits, rmode); 
      float outval = pack_float (unpack_posit(pval, m_bits, es_bits));  
      out[i] = (Eigen::half)__float2half_rn(outval);
  }
}

template <typename T>
struct PositQuantEmuFunctor <CPUDevice, T> {
  void operator()(const CPUDevice& d, int m_bits, int es_bits, int rmode, int size, const T* in, T* out) {

    QuantEmuPositCPUKernel(m_bits, es_bits, rmode, size, in, out); 
  }
};

void QuantEmuBfloat16CPUKernel(
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

  for (int i = 0; i < size; i++) {
      UFLOAT32 uf; 
      float inval = in[i];
      uf.f = inval; 
     
      unsigned int mant_grs = (uf.u & mask_mant_grs); 
      /* stochastic with 16 seeds */ 
      int seed_index = (i/16); 
      unsigned short rand = (unsigned short) xorshf_rand_with_seed(sptr_[(seed_index%16)]);
      /* stochastic rounding with 16-bit random number */ 
      uf.u += sr_mask * rand; 

      /* truncation */ 
      uf.u &= mask_mant; 

      /* round to nearest even after truncation if rne_mask is enabled */ 
      unsigned int rmask_tie = ((mant_grs & lsbGRS) >> rshift);  
      unsigned int rmask = (rmask_tie & 0x7);  
      uf.u += rne_mask * (((rmask > 0x4) || (rmask_tie == 0xC) ) << lshift); 

      out[i] = uf.f;
  }
}

void QuantEmuBfloat16CPUKernel(
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

  for (int i = 0; i < size; i++) {
      UFLOAT32 uf; 
      unsigned short *inh = (unsigned short *) in;

      float inval = __half2float(inh[i]);
      uf.f = inval; 
     
      unsigned int mant_grs = (uf.u & mask_mant_grs); 
      /* stochastic with 16 seeds */ 
      int seed_index = (i/16); 
      unsigned short rand = (unsigned short) xorshf_rand_with_seed(sptr_[(seed_index%16)]);
      /* stochastic rounding with 16-bit random number */ 
      uf.u += sr_mask * rand; 

      /* truncation */ 
      uf.u &= mask_mant; 

      /* round to nearest even after truncation if rne_mask is enabled */ 
      unsigned int rmask_tie = ((mant_grs & lsbGRS) >> rshift);  
      unsigned int rmask = (rmask_tie & 0x7);  
      uf.u += rne_mask * (((rmask > 0x4) || (rmask_tie == 0xC) ) << lshift); 

      out[i] = (Eigen::half)__float2half_rn(uf.f);
  }
}

template <typename T>
struct BfloatQuantEmuFunctor <CPUDevice, T> {
  void operator()(const CPUDevice& d, int rmode, int size, const T* in, T* out) {
    QuantEmuBfloat16CPUKernel (rmode, size, in, out); 
  }
};

void QuantEmuModFP16CPUKernel(
	const int size, 
	const float *in, 
	float *out) 
{
    #pragma omp parallel for 
    for (int i=0; i < size; i++) 
    {
      unsigned short hu = __float2half_rn(in[i]); 
      unsigned short not_denorm = ((((hu & 0x7FFF) >> 10) & 0x1F) > 0); 
      unsigned short is_denorm = (not_denorm == 0)?1:0;
       /* flush denormals to zero */ 
      hu *= !is_denorm;  
      out[i] = __half2float(hu); 
    }
}

void QuantEmuModFP16CPUKernel(
	const int size, 
	const Eigen::half *in, 
	Eigen::half *out) 
{
    #pragma omp parallel for 
    for (int i=0; i < size; i++) 
    {
      unsigned short *inptr = (unsigned short *)in; 
      unsigned short hu  = inptr[i];
      unsigned short not_denorm = ((((hu & 0x7FFF) >> 10) & 0x1F) > 0); 
      unsigned short is_denorm = (not_denorm == 0)?1:0;
      /* flush denormals to zero */ 
      hu *= !is_denorm; 
      out[i] = (Eigen::half)hu; 
    }
}


template <typename T>
struct ModFP16QuantEmuFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, const T* in, T* out) {
    QuantEmuModFP16CPUKernel(size, in, out); 
  }
};


/* OpKernel definition.
 template parameter <T> is the datatype of the tensors.*/
template <typename Device, typename T>
class QuantEmuOp : public OpKernel {
 public:
  explicit QuantEmuOp(OpKernelConstruction* context) : OpKernel(context) {
    /* aquire the OP attributes */ 
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_));
    OP_REQUIRES_OK(context, context->GetAttr("allocate_copy", &allocate_copy_));
    OP_REQUIRES_OK(context, context->GetAttr("data_type", &lpdata_type_));
    OP_REQUIRES_OK(context, context->GetAttr("precision", &mbits_));
    OP_REQUIRES_OK(context, context->GetAttr("exponent_bits", &exponent_bits_));
    OP_REQUIRES_OK(context, context->GetAttr("channel_blocking_type", &block_type_));
    OP_REQUIRES_OK(context, context->GetAttr("channels_per_block", &block_size_));
    OP_REQUIRES_OK(context, context->GetAttr("round_mode", &round_mode_));

    //std::cout << "data_format : " << data_format_ << ", lpdata_type: " << lpdata_type_ << ", mbits_: " << mbits_ << 
    // ", exponent_bits_: " << exponent_bits_ << ", quantize_grad_: " << quantize_grad_ << ", mbits_grad_ : " << mbits_grad_ << 
    // ", block_type: " << block_type_ << ", block_size: " << block_size_ << ", round_mode: " << round_mode_ << std::endl;
  }

  void Compute(OpKernelContext* context) override {
    /* Grab the input tensor */ 
    const Tensor& input_tensor = context->input(0);
    //const Tensor& nbits = context->input(1);
    //auto mbits_flat = nbits.flat<int32>(); 
    //int mbits = mbits_flat(0);
    int mbits = mbits_;
    Tensor *poutput_tensor;
    Tensor output_tensor;

    if (allocate_copy_ == 1) { 
      /* allocate fresh copy for quantized buffer */ 
      OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &poutput_tensor));
    } else { 
      /* use the same underlying buffer for both input and output tensors */  
      CHECK(output_tensor.CopyFrom(input_tensor, input_tensor.shape()));
      context->set_output(0, output_tensor);
      poutput_tensor = &output_tensor; 
      //context->set_output(0, context->input(0));
      //poutput_tensor = (Tensor*)&(context->input(0)); 
    }

    /* Do the computation. */
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    OP_REQUIRES(context, input_tensor.dims() <= 4,
                errors::InvalidArgument("Input tensor shape consists of more than 4 dimentions"));

    switch (lpdata_type_) {
      case INT: 
      case UINT: 
      {
        switch (block_type_ ) 
        {
          case NOBLOCK:
          { 
            QuantEmuFunctor<Device, T>()(
              context->eigen_device<Device>(),
              (lpdata_type_ == UINT)?1:0, 
              mbits, 	
	      round_mode_,
              static_cast<int>(input_tensor.NumElements()),
              (T*)input_tensor.flat<T>().data(),
              //output_tensor.flat<T>().data());
              poutput_tensor->flat<T>().data());
          }
          break; 
          case BLOCK_C: 
          {
	    if (data_format_.compare("channels_last")) { 
	      std::cout << "BLOCK_C blocking mode is not suported on " << data_format_ << " data format, use NHWC data_format." << std::endl; exit(0); 
            }
            if ( input_tensor.dims() > 4 ) { std::cout << "quantemu_op does not support tensors with more than 4 dimentions" << std::endl; exit (0); }
            int dims[4] = { 1, 1, 1, 1}; 
            for (int d=0; d < input_tensor.dims(); d++ ) dims[d] = input_tensor.dim_size(d); 

            BlockC_QuantEmuFunctor<Device, T>()(
	      context->eigen_device<Device>(), 
              (lpdata_type_ == UINT)?1:0, 
	      mbits, 
	      dims, 
              block_size_, 
	      round_mode_, 
              (T*)input_tensor.flat<T>().data(),
              //output_tensor.flat<T>().data());
              poutput_tensor->flat<T>().data());
          }
          break; 
          case BLOCK_CHW: 
          {
	    if (data_format_.compare("channels_first")) { 
	       std::cout << "BLOCK_CHW blocking mode is not suported on " << data_format_ << " data format, use NCHW data_format." << std::endl; exit(0); 
            }	
            if ( input_tensor.dims() > 4 ) { std::cout << "quantemu_op does not support tensors with more than 4 dimentions" << std::endl; exit (0); }
            int dims[4] = { 1, 1, 1, 1}; 
            for (int d=0; d < input_tensor.dims(); d++ ) dims[d] = input_tensor.dim_size(d); 
            //std::cout << "CHW blocking : format: " << data_format_ << ", block_size : " << block_size_ << std::endl;

            BlockCHW_QuantEmuFunctor<Device, T>()(
	      context->eigen_device<Device>(), 
              (lpdata_type_ == UINT)?1:0, 
	      mbits, 
	      dims,
              block_size_, 
	      round_mode_, 
              (T*)input_tensor.flat<T>().data(),
              //output_tensor.flat<T>().data());
              poutput_tensor->flat<T>().data());
          }
          break; 
        }
      }
      break; 
      case LOWP_FP: 
      {
	//std::cout << "LowpFP quantization called" << ", mbits : " << mbits <<  std::endl; 
        LowpFloatQuantEmuFunctor<Device, T>()(
          context->eigen_device<Device>(),
          mbits, 	
          exponent_bits_,
	  round_mode_,
          static_cast<int>(input_tensor.NumElements()),
          input_tensor.flat<T>().data(),
          //output_tensor.flat<T>().data());
          poutput_tensor->flat<T>().data());
      }
      break; 
      case LOG2: 
      {
	//std::cout << "Log2 quantization called" << ", mbits : " << mbits <<  std::endl; 
        Log2QuantEmuFunctor<Device, T>()(
          context->eigen_device<Device>(),
          static_cast<int>(input_tensor.NumElements()),
          input_tensor.flat<T>().data(),
          poutput_tensor->flat<T>().data());
      }
      break; 
      case POSIT: 
      {
	//std::cout << "Posit quantization called" << ", mbits : " << mbits <<  std::endl; 
        PositQuantEmuFunctor<Device, T>()(
          context->eigen_device<Device>(),
          mbits, 	
          exponent_bits_,
	  round_mode_,
          static_cast<int>(input_tensor.NumElements()),
          input_tensor.flat<T>().data(),
          poutput_tensor->flat<T>().data());
      }
      break; 
      case BFLOAT: 
      {
	//std::cout << "BFLOAT-16 quantization called" << ", mbits : " << mbits <<  std::endl; 
  	/* standard bfloat16 with RNE, mbits is ignored */ 
        BfloatQuantEmuFunctor<Device, T>()(
          context->eigen_device<Device>(),
	  round_mode_, 
          static_cast<int>(input_tensor.NumElements()),
          input_tensor.flat<T>().data(),
          poutput_tensor->flat<T>().data());
      }
      break; 
      case MODFP16: 
      {
        ModFP16QuantEmuFunctor<Device, T>()(
          context->eigen_device<Device>(),
          static_cast<int>(input_tensor.NumElements()),
          input_tensor.flat<T>().data(),
          poutput_tensor->flat<T>().data());
      }
      break; 
   }
  }
 private: 
  string data_format_;
  int allocate_copy_;
  int lpdata_type_;
  int mbits_;
  int exponent_bits_;
  int block_type_;   
  int block_size_;
  int round_mode_;
};

template class QuantEmuOp<CPUDevice, float>; 
template class QuantEmuOp<CPUDevice, Eigen::half>; 
/* Register the CPU kernels. */
#define REGISTER_CPU(T)                                          \
  template struct QuantEmuFunctor<CPUDevice, T>;		 \
  template struct BlockC_QuantEmuFunctor<CPUDevice, T>;		 \
  template struct BlockCHW_QuantEmuFunctor<CPUDevice, T>;	 \
  template struct LowpFloatQuantEmuFunctor<CPUDevice, T>;	 \
  template struct Log2QuantEmuFunctor<CPUDevice, T>;	         \
  template struct PositQuantEmuFunctor<CPUDevice, T>;            \
  template struct BfloatQuantEmuFunctor<CPUDevice, T>;            \
  template struct ModFP16QuantEmuFunctor<CPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("QuantizeEmu").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      QuantEmuOp<CPUDevice, T>);

REGISTER_CPU(float);
REGISTER_CPU(Eigen::half);
#ifdef GOOGLE_CUDA

template class QuantEmuOp<GPUDevice, float>; 
template class QuantEmuOp<GPUDevice, Eigen::half>; 

/* Register the GPU kernels. */ 
#define REGISTER_GPU(T)                                              \
  template struct QuantEmuFunctor<GPUDevice, T>;              	     \
  template struct BlockC_QuantEmuFunctor<GPUDevice, T>;              \
  template struct BlockCHW_QuantEmuFunctor<GPUDevice, T>;            \
  template struct LowpFloatQuantEmuFunctor<GPUDevice, T>;            \
  template struct Log2QuantEmuFunctor<GPUDevice, T>;                 \
  template struct PositQuantEmuFunctor<GPUDevice, T>;                \
  template struct BfloatQuantEmuFunctor<GPUDevice, T>;                \
  template struct ModFP16QuantEmuFunctor<GPUDevice, T>;                \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("QuantizeEmu").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      QuantEmuOp<GPUDevice, T>);

REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);

#endif /* GOOGLE_CUDA */
