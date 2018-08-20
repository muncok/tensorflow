#include "quantemu.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;  

#include <immintrin.h> 
#include <omp.h> 

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("QuantizeEmu")
    .Attr("T: {float, int32} = DT_FLOAT")
    .Input("to_quantize: T")
    .Input("mbits: int32")
    .Output("quantized: T")
    .Attr("quantize_grad: bool = False")
    .Attr("mbits_grad: int = 23")
    .Attr("lpdata_type: int = 0")
    .Attr("exponent_bits: int = 5")
    .Attr("block_type: int = 0")
    .Attr("block_size: int = 0")
    .Attr("round_mode: int = 0")
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
#endif 
#ifdef AVX 
__m256 _mm256_cvtpsph_ps(__m256 reg)
{
  __m128i hreg  = _mm256_cvtps_ph(reg, ROUNDING_CTRL);
  return  _mm256_cvtph_ps (hreg);
}
#endif 
/* CPU specialization of actual computation. */
template <typename T>
struct QuantEmuFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int mbits, T absmax, int rmode, int size, const T* in, T* out) {
    int quant_max = pow(2, mbits-1) - 1;
    T sfquant = (T)(quant_max/absmax);
    T sfdequant = (T)1.0/sfquant;
    T sfquant_br = sfquant * 4.0;  /* quantize to nbits + 2, we need them for rounding */

    __m256 vmaxpos     = _mm256_set1_ps (absmax);
    __m256 vmaxneg     = _mm256_set1_ps (-absmax);
    __m256 vsfquant    = _mm256_set1_ps (sfquant);
    __m256 vsfdequant    = _mm256_set1_ps (sfdequant);
    __m256 vsfquant_br = _mm256_set1_ps (sfquant_br);
    __m256i vrmask     = _mm256_set1_epi32 (0x3);
    __m256i vposone    = _mm256_set1_epi32 (1);
    __m256i vzero      = _mm256_set1_epi32 (0);
    __m256i vnegone    = _mm256_set1_epi32 (-1);

    __m256i vrthreshold  = _mm256_set1_epi32 (0x0); /* BIASED */
    if (rmode==NEAREST) vrthreshold  = _mm256_set1_epi32 (0x0); /* NEAREST */

    #pragma omp parallel for 
    for (int i = 0; i < size; i+=8) {
      __m256 vinp = _mm256_load_ps (&in[i]);
      /* saturate to max value */ 
      __mmask8 possatmask = _mm256_cmp_ps_mask (vinp, vmaxpos, (const int)_CMP_GT_OQ);  
      __mmask8 negsatmask = _mm256_cmp_ps_mask (vinp, vmaxneg, (const int)_CMP_LT_OQ); 

      __m256 vinpsat = _mm256_mask_mov_ps (vinp, possatmask, vmaxpos); 
      vinpsat = _mm256_mask_mov_ps (vinpsat, negsatmask, vmaxneg); 

      /* quantize */ 
      __m256i vint    = _mm256_cvtps_epi32 (_mm256_mul_ps (vinp, vsfquant));
      __m256i vint_br =_mm256_and_si256 (_mm256_cvtps_epi32 (_mm256_mul_ps (vinp, vsfquant_br)), vrmask);
      
      /* flip sign and round */ 
      __mmask8 rmask  = _mm256_cmp_epi32_mask (vint_br, vrthreshold, _MM_CMPINT_NLE);
      __m256i vsignflip = _mm256_mask_mov_epi32 (vposone, _mm256_cmp_epi32_mask (vint, vzero, _MM_CMPINT_LT), vnegone ); 
      __m256i vuint = _mm256_mul_epi32 (vint, vsignflip); 
      vuint = _mm256_mask_add_epi32 (vuint, rmask, vuint, vposone);
      vint = _mm256_mul_epi32 (vuint, vsignflip); 

      /* dequantize */ 
      __m256 vout = _mm256_mul_ps ( _mm256_cvtepi32_ps (vint), vsfdequant); 

      _mm256_store_ps (&out[i], vout); 
    }
#if 0
    #pragma openmp parallel for 
    for (int i = 0; i < size; ++i) {
      T inval = in[i];

      /* saturate anything larger than fmax_val to fmax_val */
      if (inval > absmax && inval > 0 ) inval = absmax;
      if (fabs(inval) > absmax && inval < 0 ) inval = -absmax;

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
#endif 
  }
};

/* CPU specialization of actual computation. */
template <typename T>
struct BlockQuantEmuFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int mbits, int rmode, Tensor in, Tensor out) {
    int dims[4] = { 1, 1, 1, 1}; 
    for (int d=0; d < in.dims(); d++ ) dims[d] = in.dim_size(d); 

    auto input_flat = in.flat<T>();
    auto output_flat = out.flat<T>();
    for (int d3 = 0; d3 < dims[3]; d3++) {
      for (int d2 = 0; d2 < dims[2]; d2++) {
        for (int d1 = 0; d1 < dims[1]; d1++) {
          //for (int d0 = 0; d0 < dims[0]; d0++) {
              //const int num_elements = input_flat.size();
  	      auto input_block = &input_flat(d3*dims[2]*dims[1]*dims[0] + d2*dims[1]*dims[0]+d1*dims[0]); 
  	      auto output_block = &output_flat(d3*dims[2]*dims[1]*dims[0] + d2*dims[1]*dims[0]+d1*dims[0]); 
              /* Compute absmax value for quantization */ 
              auto absfmax = 0; 
	  #pragma openmp parallel for 
              for (int i=0; i < dims[0]; i++) if (fabs(input_flat(i)) > absfmax) absfmax = fabs(input_flat(i));  
 
  	      //std::cout << " the number of elements : " << dims[3]*dims[2]*dims[1]*dims[0] << std::endl;
              QuantEmuFunctor<CPUDevice, T>()(
                d,
                mbits, 	
                absfmax,
	        rmode,
                dims[0], 
                input_block,
                output_block);
	  //}
        }
      }
    }
  }
};

template <typename T>
struct LowpFloatQuantEmuFunctor <CPUDevice, T> {
  void operator()(const CPUDevice& d, int mbits, int exp_bits, int rmode, int size, const T* in, T* out) {

    int non_mant_bits = exp_bits + 1; /* exponent + sign */
    if (mbits <= non_mant_bits) { printf("LPFP size cannot be <=5\n"); exit (0); }
    int shift = 10 - (mbits - non_mant_bits);
    unsigned short lowpfp_mask = (unsigned short)(0xFFFF << shift);
    __m128i lpmask = _mm_set1_epi16 (lowpfp_mask);

  #pragma openmp parallel for 
    for (int a= 0; a < size; a+=8) {
       __m256 v32 = _mm256_load_ps((float*)&in[a]);
       __m128i v16l = _mm256_cvtps_ph(v32, ROUNDING_CTRL);
       v16l = _mm_and_si128 (v16l, lpmask);
       v32 = _mm256_cvtph_ps (v16l);
       _mm256_store_ps((float*)&out[a], v32);
    }
  }
};

/* OpKernel definition.
 template parameter <T> is the datatype of the tensors.*/
template <typename Device, typename T>
class QuantEmuOp : public OpKernel {
 public:
  explicit QuantEmuOp(OpKernelConstruction* context) : OpKernel(context) {
    /* aquire the OP attributes */ 
    OP_REQUIRES_OK(context, context->GetAttr("quantize_grad", &quantize_grad_));
    OP_REQUIRES_OK(context, context->GetAttr("mbits_grad", &mbits_grad_));
    OP_REQUIRES_OK(context, context->GetAttr("lpdata_type", &lpdata_type_));
    OP_REQUIRES_OK(context, context->GetAttr("exponent_bits", &exponent_bits_));
    OP_REQUIRES_OK(context, context->GetAttr("block_type", &block_type_));
    OP_REQUIRES_OK(context, context->GetAttr("block_size", &block_size_));
    OP_REQUIRES_OK(context, context->GetAttr("round_mode", &round_mode_));
//    std::cout << "lpdata_type: " << lpdata_type_ << ", block_type: " << block_type_ << ", block_size: " << block_size_ << ", round_mode: " << round_mode_ << std::endl;
  }

  void Compute(OpKernelContext* context) override {
    /* Grab the input tensor */ 
    const Tensor& input_tensor = context->input(0);
    const Tensor& nbits = context->input(1);
    auto mbits_flat = nbits.flat<int32>(); 
    int mbits = mbits_flat(0);
    /* Create an output tensor */
    Tensor output_tensor;

//    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    CHECK(output_tensor.CopyFrom(input_tensor, input_tensor.shape()));
    context->set_output(0, output_tensor);
    
    /* Do the computation. */
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    OP_REQUIRES(context, input_tensor.dims() <= 4,
                errors::InvalidArgument("Input tensor shape consists of more than 4 dimentions"));
    switch (lpdata_type_) {
      case DFP_INT: 
      case DFP_UINT: 
      {
        switch (block_type_ ) 
        {
          case BLOCK_C: 
          {
            BlockQuantEmuFunctor<Device, T>()(
	      context->eigen_device<Device>(), 
	      mbits, 
	      round_mode_, 
	      input_tensor, 
	      output_tensor); 
          }
          break; 
          case NOBLOCK:
          { 
            auto input_flat = input_tensor.flat<T>();
            const int num_elements = input_flat.size();
            /* Compute absmax value for quantization */ 
            auto absfmax = 0.0; 
          #pragma openmp parallel for 
            for (int i=0; i < num_elements; i++) if (fabs(input_flat(i)) > absfmax) absfmax = fabs(input_flat(i));  
            //std::cout << "num_elements : " << num_elements  <<", absfmax val : " << absfmax << std::endl;
            QuantEmuFunctor<Device, T>()(
              context->eigen_device<Device>(),
              mbits, 	
              absfmax,
	      round_mode_,
              static_cast<int>(input_tensor.NumElements()),
              input_tensor.flat<T>().data(),
              output_tensor.flat<T>().data());
          }
          break; 
        }
      }
      break; 
      case LOWP_FP: 
      {
//	std::cout << "LowpFP quantization called" << std::endl; 
        LowpFloatQuantEmuFunctor<Device, T>()(
          context->eigen_device<Device>(),
          mbits, 	
          exponent_bits_,
	  round_mode_,
          static_cast<int>(input_tensor.NumElements()),
          input_tensor.flat<T>().data(),
          output_tensor.flat<T>().data());
      }
      break; 
    }
  }
 private: 
  bool quantize_grad_;
  int mbits_grad_; 
  int lpdata_type_;
  int exponent_bits_;
  int block_type_;   
  int block_size_;
  int round_mode_;
};

template class QuantEmuOp<CPUDevice, float>; 
//template class QuantEmuOp<CPUDevice, int32>; 
//template class QuantEmuOp<GPUDevice, float>; 
//template class QuantEmuOp<GPUDevice, int32>; 

//template struct QuantEmuFunctor<CPUDevice, int32>;
/* Register the CPU kernels. */

#define REGISTER_CPU(T)                                          \
  template struct QuantEmuFunctor<CPUDevice, T>;		 \
  template struct BlockQuantEmuFunctor<CPUDevice, T>;		 \
  template struct LowpFloatQuantEmuFunctor<CPUDevice, T>;	 \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("QuantizeEmu").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      QuantEmuOp<CPUDevice, T>);

REGISTER_CPU(float);
//REGISTER_CPU(int32);

/* Register the GPU kernels. */ 
#if 0 //def GOOGLE_CUDA

#define REGISTER_GPU(T)                                          \
  extern template struct QuantEmuFunctor<GPUDevice, T>;          \
  extern template struct BlockQuantEmuFunctor<GPUDevice, T>;     \
  extern template struct LowpFloatQuantEmuFunctor<GPUDevice, T>; \
  extern template class QuantEmuOp<GPUDevice, T>;		 \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("QuantizeEmu").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      QuantEmuOp<GPUDevice, T>);

REGISTER_GPU(float);
//REGISTER_GPU(int32);

#endif  /* GOOGLE_CUDA */

