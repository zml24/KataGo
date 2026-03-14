#ifndef NEURALNET_CUDAINCLUDES_H
#define NEURALNET_CUDAINCLUDES_H

//Ensure that CUDA_API_PER_THREAD_DEFAULT_STREAM is always defined
//before any cuda headers are included so that we get the desired threading behavior for CUDA.

#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda.h>
#include <cuda_fp16.h>
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#define KATAGO_CUDA_BFLOAT16_AVAILABLE 1
#endif

#include <cublas_v2.h>
#ifdef KATAGO_CUBLASLT_AVAILABLE
#include <cublasLt.h>
#endif
#include <cudnn.h>

#ifdef __has_include
#if __has_include(<cudnn_frontend.h>)
#include <cudnn_frontend.h>
#define KATAGO_CUDNN_FRONTEND_AVAILABLE 1
#elif __has_include(<cudnn_frontend/cudnn_frontend.h>)
#include <cudnn_frontend/cudnn_frontend.h>
#define KATAGO_CUDNN_FRONTEND_AVAILABLE 1
#endif
#endif

#if defined(KATAGO_CUDA_BFLOAT16_AVAILABLE) && defined(CUDNN_MAJOR) && CUDNN_MAJOR >= 8
#define KATAGO_CUDA_TRANSFORMER_BFLOAT16_AVAILABLE 1
#endif

#if defined(KATAGO_CUDNN_FRONTEND_AVAILABLE) && defined(CUDNN_VERSION) && CUDNN_VERSION >= 8903 && defined(CUDA_VERSION) && CUDA_VERSION >= 12000
#define KATAGO_CUDNN_SDPA_AVAILABLE 1
#endif


#endif //NEURALNET_CUDAINCLUDES_H
