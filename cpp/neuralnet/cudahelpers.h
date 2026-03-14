#ifndef NEURALNET_CUDAHELPERS_H_
#define NEURALNET_CUDAHELPERS_H_

#include "../neuralnet/cudaincludes.h"
#include "../neuralnet/activations.h"

//Given two tensors with shapes inA: [n,cA,h,w] and inB: [n,cB,h,w], that are on the GPU
//Copy them into a single tensor out: [n,cA+cB,h,w] that is also allocated on the gpu
void customCudaChannelConcat(const float* inA, const float* inB, float* out, int chwA, int chwB, int n);
void customCudaChannelConcat(const half* inA, const half* inB, half* out, int chwA, int chwB, int n);

//Given a tensor [n,c,hw], extract out channel 0 to [n,hw]
void customCudaChannel0ExtractNCHW(const float* in, float* out, int n, int c, int hw);
void customCudaChannel0ExtractNCHW(const half* in, half* out, int n, int c, int hw);
//Given a tensor [n,hw,c], extract out channel 0 to [n,hw]
void customCudaChannel0ExtractNHWC(const float* in, float* out, int n, int hw, int c);
void customCudaChannel0ExtractNHWC(const half* in, half* out, int n, int hw, int c);

//Given an input tensor and an output buffer of shape [n,c], fill output buffer with sum or max over c.
void customCudaPoolRowsSumNCHW(const float* in, float* out, int nSize, int cSize, int xySize, float scaleSum);
void customCudaPoolRowsSumNHWC(const float* in, float* out, int nSize, int xySize, int cSize, float scaleSum);

//Specialized operations for value head and general global pooling. Same as the other pooling, but fusedly fills
//an output buffer of shape [n,c*3].
void customCudaValueHeadPoolNCHW(const float* in, float* out, int nSize, int cSize, int xySize, const float* maskSum);
void customCudaValueHeadPoolNHWC(const float* in, float* out, int nSize, int xySize, int cSize, const float* maskSum);
void customCudaPoolRowsGPoolNCHW(const float* in, float* out, int nSize, int cSize, int xySize, const float* mask, const float* maskSum);
void customCudaPoolRowsGPoolNHWC(const float* in, float* out, int nSize, int xySize, int cSize, const float* mask, const float* maskSum);
void customCudaPoolRowsGPoolNCHW(const half* in, half* out, int nSize, int cSize, int xySize, const half* mask, const float* maskSum);
void customCudaPoolRowsGPoolNHWC(const half* in, half* out, int nSize, int xySize, int cSize, const half* mask, const float* maskSum);

void customCudaCopyToHalf(const float* in, half* out, int n);
void customCudaCopyFromHalf(const half* in, float* out, int n);
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
void customCudaCopyToBFloat16(const float* in, __nv_bfloat16* out, int n);
void customCudaCopyFromBFloat16(const __nv_bfloat16* in, float* out, int n);
#endif

//Given a tensor, add another tensor to it.
void customCudaAddTensorInplace(half* buf, const half* biases, int n);
//Given an input with shape [n,c] and biases of shape [c], add the biases in-place.
void customCudaAddCBiasInplaceNC(float* buf, const float* biases, int n, int c, int activation);
void customCudaAddCBiasInplaceNC(half* buf, const half* biases, int n, int c, int activation);
//Given an input with shape [n,c,xy] and biases of shape [n,c], add the biases in-place.
void customCudaAddNCBiasInplaceNCHW(float *buf, const float* biases, int nSize, int cSize, int xySize);
void customCudaAddNCBiasInplaceNCHW(half *buf, const half* biases, int nSize, int cSize, int xySize);
//Given an input with shape [n,xy,c] and biases of shape [n,c], add the biases in-place.
void customCudaAddNCBiasInplaceNHWC(float *buf, const float* biases, int nSize, int xySize, int cSize);
void customCudaAddNCBiasInplaceNHWC(half *buf, const half* biases, int nSize, int xySize, int cSize);

//Given an input with shape [n,c,xy] and scale and biases of shape [c], multiply by scale and add the biases
//Optionally also apply an activation.
//Optionally also multiply by mask (can be null), with shape [n,xy]
void customCudaApplyCScaleBiasNCHW(const float* in, float* out, const float* scale, const float* biases, const float* mask, int n, int c, int xy, int activation);
void customCudaApplyCScaleBiasNCHW(const half* in, half* out, const half* scale, const half* biases, const half* mask, int n, int c, int xy, int activation);
//Given an input with shape [n,xy,c] and scale and biases of shape [c], multiply by scale and add the biases
//Optionally also apply relu.
//Optionally also multiply by mask (can be null), with shape [n,xy]
void customCudaApplyCScaleBiasNHWC(const float* in, float* out, const float* scale, const float* biases, const float* mask, int n, int xy, int c, int activation);
void customCudaApplyCScaleBiasNHWC(const half* in, half* out, const half* scale, const half* biases, const half* mask, int n, int xy, int c, int activation);

// Transformer helpers.
void customCudaNCHWToNLCAddBiasPos(const float* spatial, const float* global, const float* pos, float* out, int n, int c, int xSize, int ySize);
void customCudaNCHWToNLCAddBiasPos(const float* spatial, const float* global, const float* pos, half* out, int n, int c, int xSize, int ySize);
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
void customCudaNCHWToNLCAddBiasPos(const float* spatial, const float* global, const float* pos, __nv_bfloat16* out, int n, int c, int xSize, int ySize);
#endif
void customCudaAddResidual(float* dst, const float* src, int n);
void customCudaAddResidual(half* dst, const half* src, int n);
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
void customCudaAddResidual(__nv_bfloat16* dst, const __nv_bfloat16* src, int n);
#endif
void customCudaRMSNorm(const float* in, float* out, const float* weight, int rows, int cSize, float eps);
void customCudaRMSNorm(const half* in, half* out, const float* weight, int rows, int cSize, float eps);
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
void customCudaRMSNorm(const __nv_bfloat16* in, __nv_bfloat16* out, const float* weight, int rows, int cSize, float eps);
#endif
void customCudaApplyRotaryInplace(float* q, float* k, const float* cos, const float* sin, int batchSize, int seqLen, int numHeads, int headDim);
void customCudaApplyRotaryInplace(half* q, half* k, const float* cos, const float* sin, int batchSize, int seqLen, int numHeads, int headDim);
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
void customCudaApplyRotaryInplace(__nv_bfloat16* q, __nv_bfloat16* k, const float* cos, const float* sin, int batchSize, int seqLen, int numHeads, int headDim);
#endif
void customCudaTransposeBSHDtoBHSD(const float* in, float* out, int batchSize, int seqLen, int numHeads, int headDim);
void customCudaTransposeBSHDtoBHSD(const half* in, half* out, int batchSize, int seqLen, int numHeads, int headDim);
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
void customCudaTransposeBSHDtoBHSD(const __nv_bfloat16* in, __nv_bfloat16* out, int batchSize, int seqLen, int numHeads, int headDim);
#endif
void customCudaTransposeBHSDtoBSHD(const float* in, float* out, int batchSize, int seqLen, int numHeads, int headDim);
void customCudaTransposeBHSDtoBSHD(const half* in, half* out, int batchSize, int seqLen, int numHeads, int headDim);
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
void customCudaTransposeBHSDtoBSHD(const __nv_bfloat16* in, __nv_bfloat16* out, int batchSize, int seqLen, int numHeads, int headDim);
#endif
void customCudaSiluMultiply(const float* a, const float* b, float* out, int n);
void customCudaSiluMultiply(const half* a, const half* b, half* out, int n);
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
void customCudaSiluMultiply(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, int n);
#endif
void customCudaMeanPoolNLC(const float* in, float* out, int batchSize, int seqLen, int cSize);
void customCudaMeanPoolNLC(const half* in, half* out, int batchSize, int seqLen, int cSize);
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
void customCudaMeanPoolNLC(const __nv_bfloat16* in, __nv_bfloat16* out, int batchSize, int seqLen, int cSize);
#endif


#endif  // NEURALNET_CUDAHELPERS_H_
