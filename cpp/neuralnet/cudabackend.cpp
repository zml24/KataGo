#ifdef USE_CUDA_BACKEND
#include "../neuralnet/cudaerrorcheck.h"
#include "../neuralnet/cudaincludes.h"

#include "../neuralnet/cudahelpers.h"
#include "../neuralnet/cudautils.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/sgfmetadata.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/transformerdesc.h"

#include "../core/simpleallocator.h"
#include "../core/test.h"

#include "../external/half-2.2.0/include/half.hpp"

#include <unordered_map>

//------------------------
#include "../core/using.h"
//------------------------

using half_t = half_float::half;
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
using bfloat16_t = __nv_bfloat16;
#endif
#ifdef KATAGO_CUDNN_SDPA_AVAILABLE
namespace fe = cudnn_frontend;
#endif

//Define this to print out some of the intermediate values of the neural net
//#define DEBUG_INTERMEDIATE_VALUES

void NeuralNet::globalInitialize() {
  //Empty for cudnn backend
}

void NeuralNet::globalCleanup() {
  cudaDeviceReset();
}

struct CudaHandles {
  cublasHandle_t cublas;
#ifdef KATAGO_CUBLASLT_AVAILABLE
  cublasLtHandle_t cublasLt;
#endif
  cudnnHandle_t cudnn;
  const int majorComputeCapability;
  const int minorComputeCapability;

  CudaHandles(int major, int minor)
    : majorComputeCapability(major),
      minorComputeCapability(minor)
  {
    CUBLAS_ERR("CudaHandles",cublasCreate(&cublas));
#ifdef KATAGO_CUBLASLT_AVAILABLE
    CUBLAS_ERR("CudaHandles",cublasLtCreate(&cublasLt));
#endif
    CUDNN_ERR("CudaHandles",cudnnCreate(&cudnn));
  }

  ~CudaHandles() {
    cublasDestroy(cublas);
#ifdef KATAGO_CUBLASLT_AVAILABLE
    cublasLtDestroy(cublasLt);
#endif
    cudnnDestroy(cudnn);
  }

  static CudaHandles* cudaHandlesTesting() {
    const int gpuIdxForThisThread = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,gpuIdxForThisThread);
    return new CudaHandles(prop.major, prop.minor);
  }

  CudaHandles(const CudaHandles&) = delete;
  CudaHandles& operator=(const CudaHandles&) = delete;
};

//---------------------------------------------------------------------------------

template<typename T>
struct ByBatchSize {
  const int maxBatchSize;
  T* data;
  cudnnStatus_t (*destroyFunc)(T);

  ByBatchSize()
    : maxBatchSize(0), data(nullptr), destroyFunc(nullptr)
  {}

  ByBatchSize(
    int maxBatchSize_
  ) : maxBatchSize(maxBatchSize_), data(nullptr), destroyFunc(nullptr) {
    data = new T[maxBatchSize];
  }

  ByBatchSize(const ByBatchSize&) = delete;
  ByBatchSize& operator=(const ByBatchSize&) = delete;

  ~ByBatchSize() {
    if(destroyFunc != nullptr && data != nullptr) {
      for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
        (*destroyFunc)(data[batchSize-1]);
      }
    }
    if(data != nullptr) {
      delete[] data;
      data = nullptr;
    }
  }
  T& operator[](int batchSize) {
    return data[batchSize-1];
  }
  const T& operator[](int batchSize) const {
    return data[batchSize-1];
  }
};

template<typename T>
struct ByBatchSizeView {
  int maxBatchSize;
  T* data;

  ByBatchSizeView()
    : maxBatchSize(0), data(nullptr)
  {}

  ByBatchSizeView(const ByBatchSize<T>& toView)
    : maxBatchSize(toView.maxBatchSize), data(toView.data)
  {}
  ByBatchSizeView& operator=(const ByBatchSize<T>& toView) {
    maxBatchSize = toView.maxBatchSize;
    data = toView.data;
  }

  ~ByBatchSizeView() {
  }
  T& operator[](int batchSize) {
    return data[batchSize-1];
  }
  const T& operator[](int batchSize) const {
    return data[batchSize-1];
  }
};

//---------------------------------------------------------------------------------


//channels, useFP16, useNHWC
typedef std::tuple<int, bool, bool> CudnnTensorDesc4DKey;

struct CudnnManager {
  const string name;
  const int maxBatchSize;
  const int nnXLen;
  const int nnYLen;
  std::map<CudnnTensorDesc4DKey, ByBatchSize<cudnnTensorDescriptor_t>*> tensorDesc4DByBatchSizeByKey;

  CudnnManager(string name_, int maxBatchSize_, int nnXLen_, int nnYLen_)
    :name(name_),
     maxBatchSize(maxBatchSize_),
     nnXLen(nnXLen_),
     nnYLen(nnYLen_),
     tensorDesc4DByBatchSizeByKey()
  {
  }

  ~CudnnManager() {
    for(auto& iter: tensorDesc4DByBatchSizeByKey) {
      delete iter.second;
    }
  }

  ByBatchSizeView<cudnnTensorDescriptor_t> getTensorDesc4DByBatchSize(
    int channels, bool useFP16, bool useNHWC
  ) {
    auto iter = tensorDesc4DByBatchSizeByKey.find({channels, useFP16, useNHWC});
    if(iter != tensorDesc4DByBatchSizeByKey.end()) {
      return ByBatchSizeView<cudnnTensorDescriptor_t>(*(iter->second));
    }
    ByBatchSize<cudnnTensorDescriptor_t>* descs = new ByBatchSize<cudnnTensorDescriptor_t>(maxBatchSize);
    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      cudnnTensorDescriptor_t& desc = (*descs)[batchSize];
      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&desc));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
                  desc,
                  (useNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
                  (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
                  batchSize,
                  channels,
                  nnYLen,
                  nnXLen
                ));
    }
    descs->destroyFunc = cudnnDestroyTensorDescriptor;
    tensorDesc4DByBatchSizeByKey[{channels, useFP16, useNHWC}] = descs;
    return ByBatchSizeView<cudnnTensorDescriptor_t>(*descs);
  }
};

//---------------------------------------------------------------------------------

struct ScratchBuffers {

  const size_t batchXYFloatBytes;
  const size_t batchFloatBytes;
  const size_t batchXYBytes;
  const size_t batchBytes;

  SimpleAllocator<void*>* allocator;

  // Not scratch, but convenient to have here
  void* zeroBuf;
  void* oneBuf;

  ScratchBuffers() = delete;
  ScratchBuffers(const ScratchBuffers&) = delete;
  ScratchBuffers& operator=(const ScratchBuffers&) = delete;

  ScratchBuffers(int maxBatchSize, int nnXLen, int nnYLen, bool useFP16)
    : batchXYFloatBytes((size_t)maxBatchSize * nnXLen * nnYLen * sizeof(float)),
      batchFloatBytes((size_t)maxBatchSize * sizeof(float)),
      batchXYBytes((size_t)maxBatchSize * nnXLen * nnYLen * (useFP16 ? sizeof(half_t) : sizeof(float))),
      batchBytes((size_t)maxBatchSize * (useFP16 ? sizeof(half_t) : sizeof(float)))
  {
    std::function<void*(size_t)> allocateFunc = [](size_t size) {
      void* buf;
      CUDA_ERR("ScratchBuffers",cudaMalloc(&buf, size));
      return buf;
    };
    std::function<void(void*)> releaseFunc = [](void* buf) {
      cudaFree(buf);
    };

    allocator = new SimpleAllocator<void*>(allocateFunc, releaseFunc);

    CudaUtils::hostMallocZeroOneBufs(zeroBuf, oneBuf, useFP16);
  }
  ~ScratchBuffers() {
    delete allocator;
    free(zeroBuf);
    free(oneBuf);
  }

  size_t getBufSizeXY(int channels) const {
    return channels * batchXYBytes;
  }
  size_t getBufSizeXYFloat(int channels) const {
    return channels * batchXYFloatBytes;
  }
  size_t getBufSizeFloat(int channels) const {
    return channels * batchFloatBytes;
  }
  size_t getBufSize(int channels) const {
    return channels * batchBytes;
  }

};


//---------------------------------------------------------------------------------

struct ConvLayer {
  const string name;
  const int inChannels;
  const int outChannels;
  ByBatchSizeView<cudnnTensorDescriptor_t> inputDescriptors;
  ByBatchSizeView<cudnnTensorDescriptor_t> outputDescriptors;
  cudnnFilterDescriptor_t filterDescriptor;
  cudnnConvolutionDescriptor_t convolutionDescriptor;
#if CUDNN_MAJOR >= 8
  ByBatchSize<cudnnConvolutionFwdAlgoPerf_t>* convolutionAlgorithms; //array of one for each batch size
#else
  ByBatchSize<cudnnConvolutionFwdAlgo_t>* convolutionAlgorithms; //array of one for each batch size
#endif
  void* filterBuf;

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;

  ConvLayer(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ConvLayerDesc* desc,
    bool useFP16,
    bool useNHWC
  ) : ConvLayer(cudaHandles, manager, desc, useFP16, useNHWC, useNHWC)
  {}

  ConvLayer(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ConvLayerDesc* desc,
    bool useFP16,
    bool useNHWCIn,
    bool useNHWCOut
  ) :
    name(desc->name),
    inChannels(desc->inChannels),
    outChannels(desc->outChannels)
  {
    int convYSize = desc->convYSize;
    int convXSize = desc->convXSize;
    int dilationY = desc->dilationY;
    int dilationX = desc->dilationX;
    int paddingX = (convXSize / 2) * dilationX;
    int paddingY = (convYSize / 2) * dilationY;

    assert(convXSize % 2 == 1);
    assert(convYSize % 2 == 1);

    inputDescriptors = manager->getTensorDesc4DByBatchSize(inChannels,useFP16,useNHWCIn);
    outputDescriptors = manager->getTensorDesc4DByBatchSize(outChannels,useFP16,useNHWCOut);
    int maxBatchSize = manager->maxBatchSize;

    bool filterNHWC = useNHWCOut && dilationY == 1 && dilationX == 1;

    CUDNN_ERR(name.c_str(),cudnnCreateFilterDescriptor(&filterDescriptor));
    CUDNN_ERR(name.c_str(),cudnnSetFilter4dDescriptor(
      filterDescriptor,
      (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
      (filterNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
      outChannels,
      inChannels,
      convYSize,
      convXSize
    ));

    int yStride = 1;
    int xStride = 1;

    //NVIDIA compute capability 7 is when we first hit Volta architecture, with tensor cores
    //See https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications
    bool tensorCoresSupported = cudaHandles->majorComputeCapability >= 7;

    CUDNN_ERR(name.c_str(),cudnnCreateConvolutionDescriptor(&convolutionDescriptor));
    CUDNN_ERR(name.c_str(),cudnnSetConvolution2dDescriptor(
      convolutionDescriptor,
      paddingY,
      paddingX,
      yStride,
      xStride,
      dilationY,
      dilationX,
      CUDNN_CROSS_CORRELATION,
      (useFP16 && !tensorCoresSupported) ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT
    ));
    if(useFP16 && tensorCoresSupported)
      CUDNN_ERR(name.c_str(),cudnnSetConvolutionMathType(convolutionDescriptor, CUDNN_TENSOR_OP_MATH));

#if CUDNN_MAJOR >= 8
    convolutionAlgorithms = new ByBatchSize<cudnnConvolutionFwdAlgoPerf_t>(maxBatchSize);
#else
    convolutionAlgorithms = new ByBatchSize<cudnnConvolutionFwdAlgo_t>(maxBatchSize);
#endif

    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      if(useFP16 && dilationX <= 1 && dilationY <= 1) {
#if CUDNN_MAJOR >= 8
        (*convolutionAlgorithms)[batchSize].algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
#else
        (*convolutionAlgorithms)[batchSize] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
#endif
      }
      else {
        const cudnnTensorDescriptor_t& inputDescriptor = inputDescriptors[batchSize];
        const cudnnTensorDescriptor_t& outputDescriptor = outputDescriptors[batchSize];

#if CUDNN_MAJOR >= 8
        int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
        int returnedAlgoCount = -1;
        cudnnConvolutionFwdAlgoPerf_t results[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
        CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardAlgorithm_v7(
          cudaHandles->cudnn,
          inputDescriptor,
          filterDescriptor,
          convolutionDescriptor,
          outputDescriptor,
          requestedAlgoCount,
          &returnedAlgoCount,
          results
        ));
        if(returnedAlgoCount <= 0)
          throw StringError("cudnnGetConvolutionForwardAlgorithm_v7 returned no algorithms?");
        (*convolutionAlgorithms)[batchSize] = results[0];
#else
        size_t bytesMemoryLimit = 0;
        CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardAlgorithm(
           cudaHandles->cudnn,
           inputDescriptor,
           filterDescriptor,
           convolutionDescriptor,
           outputDescriptor,
           CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
           bytesMemoryLimit,
           &((*convolutionAlgorithms)[batchSize])
         ));
#endif
      }
    }

    assert(desc->weights.size() == convYSize * convXSize * inChannels * outChannels);

    if(filterNHWC) {
      vector<float> weightsTransposed(desc->weights.size());
      for(int y = 0; y < convYSize; y++) {
        for(int x = 0; x < convXSize; x++) {
          for(int ic = 0; ic < inChannels; ic++) {
            for(int oc = 0; oc < outChannels; oc++) {
              weightsTransposed[((oc*convYSize + y)*convXSize + x)*inChannels + ic] =
                desc->weights[((oc*inChannels + ic)*convYSize + y)*convXSize + x];
            }
          }
        }
      }
      CudaUtils::mallocAndCopyToDevice(name,weightsTransposed,filterBuf,useFP16);
      cudaDeviceSynchronize();
    }
    else
      CudaUtils::mallocAndCopyToDevice(name,desc->weights,filterBuf,useFP16);
  }

  ~ConvLayer() {
    cudaFree(filterBuf);
    cudnnDestroyFilterDescriptor(filterDescriptor);
    cudnnDestroyConvolutionDescriptor(convolutionDescriptor);
    delete convolutionAlgorithms;
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t workspaceBytes = 0;
#if CUDNN_MAJOR >= 8
    CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardWorkspaceSize(
      cudaHandles->cudnn,
      inputDescriptors[batchSize],
      filterDescriptor,
      convolutionDescriptor,
      outputDescriptors[batchSize],
      (*convolutionAlgorithms)[batchSize].algo,
      &workspaceBytes
    ));
#else
    CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardWorkspaceSize(
      cudaHandles->cudnn,
      inputDescriptors[batchSize],
      filterDescriptor,
      convolutionDescriptor,
      outputDescriptors[batchSize],
      (*convolutionAlgorithms)[batchSize],
      &workspaceBytes
    ));
#endif
    return workspaceBytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    bool accumulate,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    const float alpha = 1.0f;
    const float beta = accumulate ? 1.0f : 0.0f;
#if CUDNN_MAJOR >= 8
    CUDNN_ERR(name.c_str(),cudnnConvolutionForward(
      cudaHandles->cudnn,
      &alpha,
      inputDescriptors[batchSize],
      inputBuf,
      filterDescriptor,
      filterBuf,
      convolutionDescriptor,
      (*convolutionAlgorithms)[batchSize].algo,
      workspaceBuf,
      workspaceBytes,
      &beta,
      outputDescriptors[batchSize],
      outputBuf
    ));
#else
    CUDNN_ERR(name.c_str(),cudnnConvolutionForward(
      cudaHandles->cudnn,
      &alpha,
      inputDescriptors[batchSize],
      inputBuf,
      filterDescriptor,
      filterBuf,
      convolutionDescriptor,
      (*convolutionAlgorithms)[batchSize],
      workspaceBuf,
      workspaceBytes,
      &beta,
      outputDescriptors[batchSize],
      outputBuf
    ));
#endif
  }

};


//---------------------------------------------------------------------------------

struct BatchNormLayer {
  const string name;
  const int numChannels;
  const float epsilon;
  const int activation;
  const int nnXLen;
  const int nnYLen;

  const bool usingFP16;
  const bool usingNHWC;

  void* mergedScaleBuf;
  void* mergedBiasBuf;

  BatchNormLayer() = delete;
  BatchNormLayer(const BatchNormLayer&) = delete;
  BatchNormLayer& operator=(const BatchNormLayer&) = delete;

  BatchNormLayer(
    CudaHandles* cudaHandles,
    const BatchNormLayerDesc* desc,
    const ActivationLayerDesc* actDesc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    numChannels(desc->numChannels),
    epsilon(desc->epsilon),
    activation(actDesc->activation),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC)
  {
    (void)cudaHandles;

    assert(desc->mean.size() == numChannels);
    assert(desc->variance.size() == numChannels);
    assert(desc->scale.size() == numChannels);
    assert(desc->bias.size() == numChannels);
    assert(desc->mergedScale.size() == numChannels);
    assert(desc->mergedBias.size() == numChannels);
    CudaUtils::mallocAndCopyToDevice(name,desc->mergedScale,mergedScaleBuf,useFP16);
    CudaUtils::mallocAndCopyToDevice(name,desc->mergedBias,mergedBiasBuf,useFP16);
  }
  ~BatchNormLayer() {
    cudaFree(mergedScaleBuf);
    cudaFree(mergedBiasBuf);
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    void* inputBuf,
    const void* maskBuf, //ok to be null
    void* outputBuf
  ) const {
    (void)cudaHandles;
    if(!usingFP16) {
      if(!usingNHWC)
        customCudaApplyCScaleBiasNCHW((const float*)inputBuf,(float*)outputBuf,(const float*)mergedScaleBuf,(const float*)mergedBiasBuf,
                                      (const float*)maskBuf,
                                      batchSize,numChannels,nnXLen*nnYLen,activation);
      else
        customCudaApplyCScaleBiasNHWC((const float*)inputBuf,(float*)outputBuf,(const float*)mergedScaleBuf,(const float*)mergedBiasBuf,
                                      (const float*)maskBuf,
                                      batchSize,nnXLen*nnYLen,numChannels,activation);
    }
    else {
      if(!usingNHWC)
        customCudaApplyCScaleBiasNCHW((const half*)inputBuf,(half*)outputBuf,(const half*)mergedScaleBuf,(const half*)mergedBiasBuf,
                                      (const half*)maskBuf,
                                      batchSize,numChannels,nnXLen*nnYLen,activation);
      else
        customCudaApplyCScaleBiasNHWC((const half*)inputBuf,(half*)outputBuf,(const half*)mergedScaleBuf,(const half*)mergedBiasBuf,
                                      (const half*)maskBuf,
                                      batchSize,nnXLen*nnYLen,numChannels,activation);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }

  }

};


//---------------------------------------------------------------------------------

struct MatMulLayer {
  const string name;
  const int inChannels;
  const int outChannels;
  const bool usingFP16;
  void* matBuf;

  MatMulLayer() = delete;
  MatMulLayer(const MatMulLayer&) = delete;
  MatMulLayer& operator=(const MatMulLayer&) = delete;

  MatMulLayer(
    CudaHandles* cudaHandles,
    const MatMulLayerDesc* desc,
    bool useFP16
  ) :
    name(desc->name),
    inChannels(desc->inChannels),
    outChannels(desc->outChannels),
    usingFP16(useFP16)
  {
    (void)cudaHandles;

    if(inChannels > 0 && outChannels > 0) {
      assert(desc->weights.size() == inChannels * outChannels);
      CudaUtils::mallocAndCopyToDevice(name,desc->weights,matBuf,useFP16);
    }
    else {
      matBuf = NULL;
    }
  }

  ~MatMulLayer() {
    if(inChannels > 0 && outChannels > 0)
      cudaFree(matBuf);
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles
  ) const {
    (void)cudaHandles;
    size_t workspaceBytes = 0;
    return workspaceBytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    (void)workspaceBuf;
    (void)workspaceBytes;
    assert(inChannels > 0 && outChannels > 0);

    if(!usingFP16) {
      const float alpha = 1.0f;
      const float beta = 0.0f;
      CUBLAS_ERR(name.c_str(),cublasSgemm(
        cudaHandles->cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        outChannels,
        batchSize,
        inChannels,
        &alpha,
        (const float*)matBuf,outChannels,
        (const float*)inputBuf,inChannels,
        &beta,
        (float*)outputBuf,outChannels
      ));
    }
    else {
      const half* alpha = (const half*)scratch->oneBuf;
      const half* beta = (const half*)scratch->zeroBuf;
      CUBLAS_ERR(name.c_str(),cublasHgemm(
        cudaHandles->cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        outChannels,
        batchSize,
        inChannels,
        alpha,
        (const half*)matBuf,outChannels,
        (const half*)inputBuf,inChannels,
        beta,
        (half*)outputBuf,outChannels
      ));
    }

  }

};

//---------------------------------------------------------------------------------

struct MatBiasLayer {
  const string name;
  const int numChannels;
  const bool usingFP16;
  const int activation;

  void* biasBuf;

  MatBiasLayer() = delete;
  MatBiasLayer(const MatBiasLayer&) = delete;
  MatBiasLayer& operator=(const MatBiasLayer&) = delete;

  MatBiasLayer(
    CudaHandles* cudaHandles,
    const MatBiasLayerDesc* desc,
    bool useFP16,
    int activation_
  ) :
    name(desc->name),
    numChannels(desc->numChannels),
    usingFP16(useFP16),
    activation(activation_)
  {
    (void)cudaHandles;
    if(numChannels > 0) {
      assert(desc->weights.size() == numChannels);
      CudaUtils::mallocAndCopyToDevice(name,desc->weights,biasBuf,useFP16);
    }
    else
      biasBuf = NULL;
  }

  ~MatBiasLayer() {
    if(numChannels > 0)
      cudaFree(biasBuf);
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    void* matBuf
  ) const {
    (void)cudaHandles;
    assert(numChannels > 0);
    if(!usingFP16) {
      customCudaAddCBiasInplaceNC((float*)matBuf,(const float*)biasBuf,batchSize,numChannels,activation);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
    else {
      customCudaAddCBiasInplaceNC((half*)matBuf,(const half*)biasBuf,batchSize,numChannels,activation);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
  }

};

//---------------------------------------------------------------------------------

struct NormActConv {
  const BatchNormLayer norm;
  const ConvLayer conv;

  const int inChannels;
  const int outChannels;
  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;

  NormActConv() = delete;
  NormActConv(const NormActConv&) = delete;
  NormActConv& operator=(const NormActConv&) = delete;

  NormActConv(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const BatchNormLayerDesc* normDesc,
    const ActivationLayerDesc* actDesc,
    const ConvLayerDesc* convDesc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ): norm(cudaHandles,normDesc,actDesc,nnX,nnY,useFP16,useNHWC),
     conv(cudaHandles,manager,convDesc,useFP16,useNHWC),
     inChannels(norm.numChannels),
     outChannels(conv.outChannels),
     nnXLen(nnX),
     nnYLen(nnY),
     usingFP16(useFP16),
     usingNHWC(useNHWC)
  {
    assert(norm.numChannels == conv.inChannels);
  }

  ~NormActConv()
  {}

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = conv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    bool accumulate,
    void* inBuf,
    void* inScratchBuf,
    void* outBuf,
    void* maskBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    norm.apply(cudaHandles,batchSize,inBuf,maskBuf,inScratchBuf);
#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint4D(string("AFTER NORM "), inScratchBuf, batchSize, inChannels, nnXLen, nnYLen, usingNHWC, usingFP16);
#endif
    conv.apply(cudaHandles,batchSize,accumulate,inScratchBuf,outBuf,workspaceBuf,workspaceBytes);
  }

};


//---------------------------------------------------------------------------------

struct ResidualBlock {
  const string name;
  const NormActConv normActConv1;
  const NormActConv normActConv2;

  ResidualBlock() = delete;
  ResidualBlock(const ResidualBlock&) = delete;
  ResidualBlock& operator=(const ResidualBlock&) = delete;

  ResidualBlock(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ResidualBlockDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ): name(desc->name),
     normActConv1(cudaHandles,manager,&desc->preBN,&desc->preActivation,&desc->regularConv,nnX,nnY,useFP16,useNHWC),
     normActConv2(cudaHandles,manager,&desc->midBN,&desc->midActivation,&desc->finalConv,nnX,nnY,useFP16,useNHWC)
  {
  }

  ~ResidualBlock()
  {}

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = normActConv1.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = normActConv2.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> midIn(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    SizedBuf<void*> midScratch(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    normActConv1.apply(cudaHandles,batchSize,false,trunkBuf,trunkScratchBuf,midIn.buf,maskBuf,workspaceBuf,workspaceBytes);
    normActConv2.apply(cudaHandles,batchSize,true,midIn.buf,midScratch.buf,trunkBuf,maskBuf,workspaceBuf,workspaceBytes);
  }

};


//----------------------------------------------------------------------------


struct GlobalPoolingResidualBlock {
  const string name;
  const BatchNormLayer preBN;
  const ConvLayer regularConv;
  const ConvLayer gpoolConv;
  const BatchNormLayer gpoolBN;
  const MatMulLayer gpoolToBiasMul;
  const NormActConv normActConv2;

  const int nnXLen;
  const int nnYLen;
  const int regularChannels;
  const int gpoolChannels;
  const bool usingFP16;
  const bool usingNHWC;

  GlobalPoolingResidualBlock() = delete;
  GlobalPoolingResidualBlock(const GlobalPoolingResidualBlock&) = delete;
  GlobalPoolingResidualBlock& operator=(const GlobalPoolingResidualBlock&) = delete;

  GlobalPoolingResidualBlock(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const GlobalPoolingResidualBlockDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ): name(desc->name),
     preBN(cudaHandles,&desc->preBN,&desc->preActivation,nnX,nnY,useFP16,useNHWC),
     regularConv(cudaHandles,manager,&desc->regularConv,useFP16,useNHWC),
     gpoolConv(cudaHandles,manager,&desc->gpoolConv,useFP16,useNHWC),
     gpoolBN(cudaHandles,&desc->gpoolBN,&desc->gpoolActivation,nnX,nnY,useFP16,useNHWC),
     gpoolToBiasMul(cudaHandles,&desc->gpoolToBiasMul,useFP16),
     normActConv2(cudaHandles,manager,&desc->midBN,&desc->midActivation,&desc->finalConv,nnX,nnY,useFP16,useNHWC),
     nnXLen(nnX),
     nnYLen(nnY),
     regularChannels(desc->regularConv.outChannels),
     gpoolChannels(desc->gpoolConv.outChannels),
     usingFP16(useFP16),
     usingNHWC(useNHWC)
  {
  }

  ~GlobalPoolingResidualBlock() {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = regularConv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolConv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolToBiasMul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = normActConv2.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = sizeof(float)*batchSize*gpoolChannels*nnXLen*nnYLen;
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> regularOut(scratch->allocator, scratch->getBufSizeXY(regularChannels));
    SizedBuf<void*> regularScratch(scratch->allocator, scratch->getBufSizeXY(regularChannels));
    SizedBuf<void*> gpoolOut(scratch->allocator, scratch->getBufSizeXY(gpoolChannels));
    SizedBuf<void*> gpoolOut2(scratch->allocator, scratch->getBufSizeXY(gpoolChannels));
    SizedBuf<void*> gpoolConcat(scratch->allocator, scratch->getBufSize(gpoolChannels*3));
    SizedBuf<void*> gpoolBias(scratch->allocator, scratch->getBufSize(regularChannels));

    preBN.apply(cudaHandles,batchSize,trunkBuf,maskBuf,trunkScratchBuf);
    regularConv.apply(cudaHandles,batchSize,false,trunkScratchBuf,regularOut.buf,workspaceBuf,workspaceBytes);
    gpoolConv.apply(cudaHandles,batchSize,false,trunkScratchBuf,gpoolOut.buf,workspaceBuf,workspaceBytes);
    gpoolBN.apply(cudaHandles,batchSize,gpoolOut.buf,maskBuf,gpoolOut2.buf);

    if(!usingFP16) {
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const float*)gpoolOut2.buf,(float*)gpoolConcat.buf,batchSize,gpoolChannels,nnXLen*nnYLen,(const float*)maskBuf,maskSumBuf);
      else
        customCudaPoolRowsGPoolNHWC((const float*)gpoolOut2.buf,(float*)gpoolConcat.buf,batchSize,nnXLen*nnYLen,gpoolChannels,(const float*)maskBuf,maskSumBuf);
    }
    else {
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const half*)gpoolOut2.buf,(half*)gpoolConcat.buf,batchSize,gpoolChannels,nnXLen*nnYLen,(const half*)maskBuf,maskSumBuf);
      else
        customCudaPoolRowsGPoolNHWC((const half*)gpoolOut2.buf,(half*)gpoolConcat.buf,batchSize,nnXLen*nnYLen,gpoolChannels,(const half*)maskBuf,maskSumBuf);
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    gpoolToBiasMul.apply(cudaHandles,scratch,batchSize,gpoolConcat.buf,gpoolBias.buf,workspaceBuf,workspaceBytes);

    if(!usingFP16) {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((float*)regularOut.buf,(const float*)gpoolBias.buf,batchSize,regularChannels,nnXLen*nnYLen);
      else
        customCudaAddNCBiasInplaceNHWC((float*)regularOut.buf,(const float*)gpoolBias.buf,batchSize,nnXLen*nnYLen,regularChannels);
    }
    else {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((half*)regularOut.buf,(const half*)gpoolBias.buf,batchSize,regularChannels,nnXLen*nnYLen);
      else
        customCudaAddNCBiasInplaceNHWC((half*)regularOut.buf,(const half*)gpoolBias.buf,batchSize,nnXLen*nnYLen,regularChannels);
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    normActConv2.apply(cudaHandles,batchSize,true,regularOut.buf,regularScratch.buf,trunkBuf,maskBuf,workspaceBuf,workspaceBytes);
  }

};

//------------------------------------------------------------------------------

struct BlockStack {
  const int numBlocks;
  const int trunkNumChannels;
  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;
  vector<pair<int,unique_ptr_void>> blocks;

  BlockStack() = delete;
  BlockStack(const BlockStack&) = delete;
  BlockStack& operator=(const BlockStack&) = delete;

  BlockStack(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    int nBlocks,
    int trunkChannels,
    const std::vector<std::pair<int, unique_ptr_void>>& descBlocks,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  );
  ~BlockStack();

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const;

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* maskBuf,
    float* maskSumBuf,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const;

};

//------------------------------------------------------------------------------

struct NestedBottleneckResidualBlock {
  const string name;
  const NormActConv normActConv1;
  const BlockStack blocks;
  const NormActConv normActConv2;

  NestedBottleneckResidualBlock() = delete;
  NestedBottleneckResidualBlock(const NestedBottleneckResidualBlock&) = delete;
  NestedBottleneckResidualBlock& operator=(const NestedBottleneckResidualBlock&) = delete;

  NestedBottleneckResidualBlock(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const NestedBottleneckResidualBlockDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ): name(desc->name),
     normActConv1(cudaHandles,manager,&desc->preBN,&desc->preActivation,&desc->preConv,nnX,nnY,useFP16,useNHWC),
     blocks(cudaHandles,manager,desc->numBlocks,desc->preConv.outChannels,desc->blocks,nnX,nnY,useFP16,useNHWC),
     normActConv2(cudaHandles,manager,&desc->postBN,&desc->postActivation,&desc->postConv,nnX,nnY,useFP16,useNHWC)
  {
  }

  ~NestedBottleneckResidualBlock()
  {}

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = normActConv1.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = blocks.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = normActConv2.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> mid(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    SizedBuf<void*> midScratch(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    assert(normActConv1.outChannels == normActConv2.inChannels);
    normActConv1.apply(cudaHandles,batchSize,false,trunkBuf,trunkScratchBuf,mid.buf,maskBuf,workspaceBuf,workspaceBytes);
    blocks.apply(
      cudaHandles,
      scratch,
      batchSize,
      maskBuf,
      maskSumBuf,
      mid.buf,
      midScratch.buf,
      workspaceBuf,
      workspaceBytes
    );
    normActConv2.apply(cudaHandles,batchSize,true,mid.buf,midScratch.buf,trunkBuf,maskBuf,workspaceBuf,workspaceBytes);
  }

};

//------------------------------------------------------------------------------

BlockStack::BlockStack(
  CudaHandles* cudaHandles,
  CudnnManager* manager,
  int nBlocks,
  int trunkChannels,
  const std::vector<std::pair<int, unique_ptr_void>>& descBlocks,
  int nnX,
  int nnY,
  bool useFP16,
  bool useNHWC
) :
  numBlocks(nBlocks),
  trunkNumChannels(trunkChannels),
  nnXLen(nnX),
  nnYLen(nnY),
  usingFP16(useFP16),
  usingNHWC(useNHWC)
{
  assert(numBlocks == descBlocks.size());
  for(int i = 0; i<numBlocks; i++) {
    if(descBlocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlockDesc* blockDesc = (ResidualBlockDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new ResidualBlock(
          cudaHandles,
          manager,
          blockDesc,
          nnXLen,
          nnYLen,
          useFP16,
          useNHWC
        )
      );
      blocks.push_back(make_pair(ORDINARY_BLOCK_KIND,std::move(blockPtr)));
    }
    else if(descBlocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlockDesc* blockDesc = (GlobalPoolingResidualBlockDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new GlobalPoolingResidualBlock(
          cudaHandles,
          manager,
          blockDesc,
          nnXLen,
          nnYLen,
          useFP16,
          useNHWC
        )
      );
      blocks.push_back(make_pair(GLOBAL_POOLING_BLOCK_KIND,std::move(blockPtr)));
    }
    else if(descBlocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlockDesc* blockDesc = (NestedBottleneckResidualBlockDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new NestedBottleneckResidualBlock(
          cudaHandles,
          manager,
          blockDesc,
          nnXLen,
          nnYLen,
          useFP16,
          useNHWC
        )
      );
      blocks.push_back(make_pair(NESTED_BOTTLENECK_BLOCK_KIND,std::move(blockPtr)));
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
}
BlockStack::~BlockStack() {
}

size_t BlockStack::requiredWorkspaceBytes(
  CudaHandles* cudaHandles,
  int batchSize
) const {
  size_t bytes = 0;
  size_t b;

  for(int i = 0; i<blocks.size(); i++) {
    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlock* block = (ResidualBlock*)blocks[i].second.get();
      b = block->requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlock* block = (GlobalPoolingResidualBlock*)blocks[i].second.get();
      b = block->requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlock* block = (NestedBottleneckResidualBlock*)blocks[i].second.get();
      b = block->requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
  return bytes;
}

void BlockStack::apply(
  CudaHandles* cudaHandles,
  ScratchBuffers* scratch,
  int batchSize,
  void* maskBuf,
  float* maskSumBuf,
  void* trunkBuf,
  void* trunkScratchBuf,
  void* workspaceBuf,
  size_t workspaceBytes
) const {

  for(int i = 0; i<blocks.size(); i++) {
#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint4D(string("Blockstack before block " + Global::intToString(i)), trunkBuf, batchSize, trunkNumChannels, nnXLen, nnYLen, usingNHWC, usingFP16);
#endif

    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlock* block = (ResidualBlock*)blocks[i].second.get();
      block->apply(
        cudaHandles,
        scratch,
        batchSize,
        trunkBuf,
        trunkScratchBuf,
        maskBuf,
        workspaceBuf,
        workspaceBytes
      );
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlock* block = (GlobalPoolingResidualBlock*)blocks[i].second.get();
      block->apply(
        cudaHandles,
        scratch,
        batchSize,
        trunkBuf,
        trunkScratchBuf,
        maskBuf,
        maskSumBuf,
        workspaceBuf,
        workspaceBytes
      );
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlock* block = (NestedBottleneckResidualBlock*)blocks[i].second.get();
      block->apply(
        cudaHandles,
        scratch,
        batchSize,
        trunkBuf,
        trunkScratchBuf,
        maskBuf,
        maskSumBuf,
        workspaceBuf,
        workspaceBytes
      );
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
}
//------------------------------------------------------------------------------

struct SGFMetadataEncoder {
  const string name;

  const bool usingFP16;

  const MatMulLayer mul1;
  const MatBiasLayer bias1;
  const MatMulLayer mul2;
  const MatBiasLayer bias2;
  const MatMulLayer mul3;

  SGFMetadataEncoder() = delete;
  SGFMetadataEncoder(const SGFMetadataEncoder&) = delete;
  SGFMetadataEncoder& operator=(const SGFMetadataEncoder&) = delete;

  SGFMetadataEncoder(
    CudaHandles* cudaHandles,
    const SGFMetadataEncoderDesc* desc,
    bool useFP16
  ) :
    name(desc->name),
    usingFP16(useFP16),
    mul1(cudaHandles,&desc->mul1,useFP16),
    bias1(cudaHandles,&desc->bias1,useFP16,desc->act1.activation),
    mul2(cudaHandles,&desc->mul2,useFP16),
    bias2(cudaHandles,&desc->bias2,useFP16,desc->act2.activation),
    mul3(cudaHandles,&desc->mul3,useFP16)
  {
  }

  ~SGFMetadataEncoder()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    (void)batchSize;
    size_t bytes = 0;
    size_t b;

    b = mul1.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = mul2.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = mul3.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);

    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> internalBuf1(scratch->allocator, scratch->getBufSizeFloat(std::max(mul1.outChannels,mul2.outChannels)));
    SizedBuf<void*> internalBuf2(scratch->allocator, scratch->getBufSizeFloat(std::max(mul1.outChannels,mul2.outChannels)));

    mul1.apply(cudaHandles,scratch,batchSize,inputBuf,internalBuf1.buf,workspaceBuf,workspaceBytes);
    bias1.apply(cudaHandles,batchSize,internalBuf1.buf);
    mul2.apply(cudaHandles,scratch,batchSize,internalBuf1.buf,internalBuf2.buf,workspaceBuf,workspaceBytes);
    bias2.apply(cudaHandles,batchSize,internalBuf2.buf);
    mul3.apply(cudaHandles,scratch,batchSize,internalBuf2.buf,outputBuf,workspaceBuf,workspaceBytes);
  }

};


//----------------------------------------------------------------------------

struct Trunk {
  const string name;
  const int modelVersion;
  const int numBlocks;
  const int trunkNumChannels;

  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;

  std::unique_ptr<ConvLayer> initialConv;
  std::unique_ptr<MatMulLayer> initialMatMul;
  std::unique_ptr<SGFMetadataEncoder> sgfMetadataEncoder;
  const BlockStack blocks;
  std::unique_ptr<BatchNormLayer> trunkTipBN;

  Trunk() = delete;
  Trunk(const Trunk&) = delete;
  Trunk& operator=(const Trunk&) = delete;

  Trunk(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const TrunkDesc* desc,
    int nnX,
    int nnY,
    bool inputsUseNHWC,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    modelVersion(desc->modelVersion),
    numBlocks(desc->numBlocks),
    trunkNumChannels(desc->trunkNumChannels),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    blocks(cudaHandles,manager,desc->numBlocks,desc->trunkNumChannels,desc->blocks,nnX,nnY,useFP16,useNHWC)
  {
    int midNumChannels = desc->midNumChannels;
    int regularNumChannels = desc->regularNumChannels;
    int gpoolNumChannels = desc->gpoolNumChannels;

    int maxBatchSize = manager->maxBatchSize;
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,trunkNumChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,midNumChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,regularNumChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,gpoolNumChannels);

    initialConv = std::make_unique<ConvLayer>(cudaHandles,manager,&desc->initialConv,useFP16,inputsUseNHWC,useNHWC);
    initialMatMul = std::make_unique<MatMulLayer>(cudaHandles,&desc->initialMatMul,useFP16);
    if(desc->metaEncoderVersion > 0) {
      sgfMetadataEncoder = std::make_unique<SGFMetadataEncoder>(cudaHandles,&desc->sgfMetadataEncoder,useFP16);
      testAssert(sgfMetadataEncoder->mul3.outChannels == initialMatMul->outChannels);
    }

    trunkTipBN = std::make_unique<BatchNormLayer>(cudaHandles,&desc->trunkTipBN,&desc->trunkTipActivation,nnXLen,nnYLen,useFP16,useNHWC);
    assert(desc->blocks.size() == numBlocks);
  }

  ~Trunk()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    b = initialConv->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);

    b = initialMatMul->requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);

    if(sgfMetadataEncoder != nullptr) {
      b = sgfMetadataEncoder->requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }

    b = blocks.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* inputBuf,
    void* inputGlobalBuf,
    void* inputMetaBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* trunkBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {

    SizedBuf<void*> trunkScratch(scratch->allocator, scratch->getBufSizeXY(trunkNumChannels));

    //Feed the conv into trunkScratch.buf, not trunkBuf
    initialConv->apply(cudaHandles,batchSize,false,inputBuf,trunkScratch.buf,workspaceBuf,workspaceBytes);

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint4D(string("After initial conv"), trunkScratch.buf, batchSize, trunkNumChannels, nnXLen, nnYLen, usingNHWC, usingFP16);
    #endif

    //Feed the matmul into trunkBuf
    initialMatMul->apply(cudaHandles,scratch,batchSize,inputGlobalBuf,trunkBuf,workspaceBuf,workspaceBytes);
    //Then accumulate it into trunkScratch.buf, broadcasting during the process
    if(!usingFP16) {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((float*)trunkScratch.buf,(const float*)trunkBuf,batchSize,trunkNumChannels,nnXLen*nnYLen);
      else
        customCudaAddNCBiasInplaceNHWC((float*)trunkScratch.buf,(const float*)trunkBuf,batchSize,nnXLen*nnYLen,trunkNumChannels);
    }
    else {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((half*)trunkScratch.buf,(const half*)trunkBuf,batchSize,trunkNumChannels,nnXLen*nnYLen);
      else
        customCudaAddNCBiasInplaceNHWC((half*)trunkScratch.buf,(const half*)trunkBuf,batchSize,nnXLen*nnYLen,trunkNumChannels);
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    if(sgfMetadataEncoder != nullptr) {
      testAssert(inputMetaBuf != NULL);
      //Feed the result into trunkBuf
      sgfMetadataEncoder->apply(cudaHandles,scratch,batchSize,inputMetaBuf,trunkBuf,workspaceBuf,workspaceBytes);
      //Then accumulate it into trunkScratch.buf, broadcasting during the process
      if(!usingFP16) {
        if(!usingNHWC)
          customCudaAddNCBiasInplaceNCHW((float*)trunkScratch.buf,(const float*)trunkBuf,batchSize,trunkNumChannels,nnXLen*nnYLen);
        else
          customCudaAddNCBiasInplaceNHWC((float*)trunkScratch.buf,(const float*)trunkBuf,batchSize,nnXLen*nnYLen,trunkNumChannels);
      }
      else {
        if(!usingNHWC)
          customCudaAddNCBiasInplaceNCHW((half*)trunkScratch.buf,(const half*)trunkBuf,batchSize,trunkNumChannels,nnXLen*nnYLen);
        else
          customCudaAddNCBiasInplaceNHWC((half*)trunkScratch.buf,(const half*)trunkBuf,batchSize,nnXLen*nnYLen,trunkNumChannels);
      }
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
    else {
      testAssert(inputMetaBuf == NULL);
    }

    //Flip trunkBuf and trunkScratch.buf so that the result gets accumulated in trunkScratch.buf
    blocks.apply(
      cudaHandles,
      scratch,
      batchSize,
      maskBuf,
      maskSumBuf,
      trunkScratch.buf,
      trunkBuf,
      workspaceBuf,
      workspaceBytes
    );

    //And now with the final BN port it from trunkScratch.buf to trunkBuf.
    trunkTipBN->apply(cudaHandles,batchSize,trunkScratch.buf,maskBuf,trunkBuf);

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint4D(string("Trunk tip"), trunkBuf, batchSize, trunkNumChannels, nnXLen, nnYLen, usingNHWC, usingFP16);
    #endif
  }

};

//------------------------------------------------------------------------------

static void fillMaskFloatBufAndMaskSumBuf(void* maskBuf, float*& maskFloatBuf, float*& maskSumBuf, bool usingFP16, int batchSize, int nnXLen, int nnYLen) {
  if(!usingFP16) {
    maskFloatBuf = (float*)maskBuf;
    customCudaPoolRowsSumNCHW((const float*)maskFloatBuf,maskSumBuf,batchSize,1,nnXLen*nnYLen,1.0);
    CUDA_ERR("sumMask",cudaPeekAtLastError());
  }
  else {
    customCudaCopyFromHalf((const half*)maskBuf,maskFloatBuf,batchSize*nnXLen*nnYLen);
    CUDA_ERR("copyMaskFromHalf",cudaPeekAtLastError());
    customCudaPoolRowsSumNCHW((const float*)maskFloatBuf,maskSumBuf,batchSize,1,nnXLen*nnYLen,1.0);
    CUDA_ERR("sumMask",cudaPeekAtLastError());
  }
}


//------------------------------------------------------------------------------

struct PolicyHead {
  const string name;
  const int modelVersion;
  const int nnXLen;
  const int nnYLen;
  const int p1Channels;
  const int g1Channels;
  const int p2Channels;
  const bool usingFP16;
  const bool usingNHWC;

  const ConvLayer p1Conv;
  const ConvLayer g1Conv;
  const BatchNormLayer g1BN;
  const MatMulLayer gpoolToBiasMul;
  const BatchNormLayer p1BN;
  const ConvLayer p2Conv;
  const MatMulLayer gpoolToPassMul;
  const MatBiasLayer gpoolToPassBias;
  const MatMulLayer gpoolToPassMul2;

  PolicyHead() = delete;
  PolicyHead(const PolicyHead&) = delete;
  PolicyHead& operator=(const PolicyHead&) = delete;

  PolicyHead(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const PolicyHeadDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    modelVersion(desc->modelVersion),
    nnXLen(nnX),
    nnYLen(nnY),
    p1Channels(desc->p1Conv.outChannels),
    g1Channels(desc->g1Conv.outChannels),
    p2Channels(desc->p2Conv.outChannels),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    p1Conv(cudaHandles,manager,&desc->p1Conv,useFP16,useNHWC),
    g1Conv(cudaHandles,manager,&desc->g1Conv,useFP16,useNHWC),
    g1BN(cudaHandles,&desc->g1BN,&desc->g1Activation,nnX,nnY,useFP16,useNHWC),
    gpoolToBiasMul(cudaHandles,&desc->gpoolToBiasMul,false),
    p1BN(cudaHandles,&desc->p1BN,&desc->p1Activation,nnX,nnY,false,useNHWC),
    p2Conv(cudaHandles,manager,&desc->p2Conv,false,useNHWC),
    gpoolToPassMul(cudaHandles,&desc->gpoolToPassMul,false),
    gpoolToPassBias(cudaHandles,&desc->gpoolToPassBias,false,desc->passActivation.activation),
    gpoolToPassMul2(cudaHandles,&desc->gpoolToPassMul2,false)
  {
  }

  ~PolicyHead()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    b = p1Conv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = g1Conv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolToBiasMul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = p2Conv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolToPassMul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = gpoolToPassMul2.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = sizeof(float)*batchSize*g1Channels*nnXLen*nnYLen;
    bytes = std::max(bytes,b);

    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* maskBuf,
    float* maskFloatBuf,
    float* maskSumBuf,
    void* trunkBuf,
    float* policyPassBuf,
    float* policyBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {

    SizedBuf<void*> p1Out(scratch->allocator, scratch->getBufSizeXYFloat(p1Channels)); //Need to hold floats, not just halfs
    SizedBuf<void*> p1Out2(scratch->allocator, scratch->getBufSizeXYFloat(p1Channels)); //Need to hold floats, not just halfs
    SizedBuf<void*> g1Out(scratch->allocator, scratch->getBufSizeXY(g1Channels));
    SizedBuf<void*> g1Out2(scratch->allocator, scratch->getBufSizeXY(g1Channels));
    SizedBuf<void*> g1Concat(scratch->allocator, scratch->getBufSizeFloat(g1Channels*3));
    SizedBuf<void*> g1Bias(scratch->allocator, scratch->getBufSizeFloat(p1Channels));
    SizedBuf<void*> p1Pass(scratch->allocator, scratch->getBufSizeFloat(p1Channels));

    p1Conv.apply(cudaHandles,batchSize,false,trunkBuf,p1Out.buf,workspaceBuf,workspaceBytes);
    g1Conv.apply(cudaHandles,batchSize,false,trunkBuf,g1Out.buf,workspaceBuf,workspaceBytes);
    g1BN.apply(cudaHandles,batchSize,g1Out.buf,maskBuf,g1Out2.buf);

    if(!usingFP16) {
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const float*)g1Out2.buf,(float*)g1Concat.buf,batchSize,g1Channels,nnXLen*nnYLen,maskFloatBuf,maskSumBuf);
      else
        customCudaPoolRowsGPoolNHWC((const float*)g1Out2.buf,(float*)g1Concat.buf,batchSize,nnXLen*nnYLen,g1Channels,maskFloatBuf,maskSumBuf);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
    else {
      customCudaCopyFromHalf((const half*)g1Out2.buf,(float*)workspaceBuf,batchSize*g1Channels*nnXLen*nnYLen);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const float*)workspaceBuf,(float*)g1Concat.buf,batchSize,g1Channels,nnXLen*nnYLen,maskFloatBuf,maskSumBuf);
      else
        customCudaPoolRowsGPoolNHWC((const float*)workspaceBuf,(float*)g1Concat.buf,batchSize,nnXLen*nnYLen,g1Channels,maskFloatBuf,maskSumBuf);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }

    gpoolToBiasMul.apply(cudaHandles,scratch,batchSize,g1Concat.buf,g1Bias.buf,workspaceBuf,workspaceBytes);

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint4D(string("p1 pre-gpool-sum"), p1Out.buf, batchSize, p1Channels, nnXLen, nnYLen, usingNHWC, usingFP16);
    CudaUtils::debugPrint4D(string("g1 pre-gpool"), g1Out.buf, batchSize, g1Channels, nnXLen, nnYLen, usingNHWC, usingFP16);
    CudaUtils::debugPrint2D(string("g1 pooled"), g1Concat.buf, batchSize, g1Channels*3, false);
    CudaUtils::debugPrint2D(string("g1 biases"), g1Bias.buf, batchSize, p1Channels, false);
    #endif

    float* p1OutBufA;
    float* p1OutBufB;
    if(!usingFP16) {
      p1OutBufA = (float*)p1Out.buf;
      p1OutBufB = (float*)p1Out2.buf;
    }
    else {
      customCudaCopyFromHalf((const half*)p1Out.buf,(float*)p1Out2.buf,batchSize*p1Channels*nnXLen*nnYLen);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      p1OutBufA = (float*)p1Out2.buf;
      p1OutBufB = (float*)p1Out.buf;
    }

    if(!usingNHWC)
      customCudaAddNCBiasInplaceNCHW(p1OutBufA,(float*)g1Bias.buf,batchSize,p1Channels,nnXLen*nnYLen);
    else
      customCudaAddNCBiasInplaceNHWC(p1OutBufA,(float*)g1Bias.buf,batchSize,nnXLen*nnYLen,p1Channels);
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    p1BN.apply(cudaHandles,batchSize,p1OutBufA,maskFloatBuf,p1OutBufB);
    p2Conv.apply(cudaHandles,batchSize,false,p1OutBufB,(float*)policyBuf,workspaceBuf,workspaceBytes);

    if(modelVersion >= 15) {
      gpoolToPassMul.apply(cudaHandles,scratch,batchSize,g1Concat.buf,p1Pass.buf,workspaceBuf,workspaceBytes);
      gpoolToPassBias.apply(cudaHandles,batchSize,p1Pass.buf);
      gpoolToPassMul2.apply(cudaHandles,scratch,batchSize,p1Pass.buf,policyPassBuf,workspaceBuf,workspaceBytes);
    }
    else {
      gpoolToPassMul.apply(cudaHandles,scratch,batchSize,g1Concat.buf,policyPassBuf,workspaceBuf,workspaceBytes);
    }

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint4D(string("p1 after-gpool-sum"), p1OutBufA, batchSize, p1Channels, nnXLen, nnYLen, usingNHWC, false);
    CudaUtils::debugPrint2D(string("policypass"), policyPassBuf, batchSize, 1, false);
    CudaUtils::debugPrint4D(string("policy"), policyBuf, batchSize, p2Channels, nnXLen, nnYLen, usingNHWC, false);
    #endif

  }

};

//------------------------------------------------------------------------------

struct ValueHead {
  const string name;
  const int modelVersion;
  const int nnXLen;
  const int nnYLen;
  const int v1Channels;
  const int v2Channels;
  const int valueChannels;
  const int scoreValueChannels;
  const int ownershipChannels;
  const bool usingFP16;
  const bool usingNHWC;

  const ConvLayer v1Conv;
  const BatchNormLayer v1BN;
  const MatMulLayer v2Mul;
  const MatBiasLayer v2Bias;
  const MatMulLayer v3Mul;
  const MatBiasLayer v3Bias;
  const MatMulLayer sv3Mul;
  const MatBiasLayer sv3Bias;
  const ConvLayer vOwnershipConv;

  ValueHead() = delete;
  ValueHead(const ValueHead&) = delete;
  ValueHead& operator=(const ValueHead&) = delete;

  ValueHead(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ValueHeadDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    modelVersion(desc->modelVersion),
    nnXLen(nnX),
    nnYLen(nnY),
    v1Channels(desc->v1Conv.outChannels),
    v2Channels(desc->v2Mul.outChannels),
    valueChannels(desc->v3Mul.outChannels),
    scoreValueChannels(desc->sv3Mul.outChannels),
    ownershipChannels(desc->vOwnershipConv.outChannels),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    v1Conv(cudaHandles,manager,&desc->v1Conv,useFP16,useNHWC),
    v1BN(cudaHandles,&desc->v1BN,&desc->v1Activation,nnX,nnY,useFP16,useNHWC),
    v2Mul(cudaHandles,&desc->v2Mul,false),
    v2Bias(cudaHandles,&desc->v2Bias,false,desc->v2Activation.activation),
    v3Mul(cudaHandles,&desc->v3Mul,false),
    v3Bias(cudaHandles,&desc->v3Bias,false,ACTIVATION_IDENTITY),
    sv3Mul(cudaHandles,&desc->sv3Mul,false),
    sv3Bias(cudaHandles,&desc->sv3Bias,false,ACTIVATION_IDENTITY),
    vOwnershipConv(cudaHandles,manager,&desc->vOwnershipConv,useFP16,useNHWC)
  {
  }

  ~ValueHead()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    b = v1Conv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = v2Mul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = v3Mul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = sizeof(float)*batchSize*v1Channels*nnXLen*nnYLen;
    bytes = std::max(bytes,b);

    b = sv3Mul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = vOwnershipConv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = sizeof(float)*batchSize*ownershipChannels*nnXLen*nnYLen;
    bytes = std::max(bytes,b);

    return bytes;
  }


  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* maskBuf,
    float* maskSumBuf,
    void* trunkBuf,
    float* valueBuf,
    float* scoreValueBuf,
    void* ownershipBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> v1Out(scratch->allocator, scratch->getBufSizeXY(v1Channels));
    SizedBuf<void*> v1Out2(scratch->allocator, scratch->getBufSizeXY(v1Channels));
    SizedBuf<void*> v1Mean(scratch->allocator, scratch->getBufSizeFloat(v1Channels*3));
    SizedBuf<void*> v2Out(scratch->allocator, scratch->getBufSizeFloat(v2Channels));
    SizedBuf<void*> ownershipScratch(scratch->allocator, scratch->getBufSizeXYFloat(ownershipChannels));

    v1Conv.apply(cudaHandles,batchSize,false,trunkBuf,v1Out.buf,workspaceBuf,workspaceBytes);
    v1BN.apply(cudaHandles,batchSize,v1Out.buf,maskBuf,v1Out2.buf);

    void* bufToBePooled = v1Out2.buf;
    if(usingFP16) {
      customCudaCopyFromHalf((const half*)v1Out2.buf,(float*)workspaceBuf,batchSize*v1Channels*nnXLen*nnYLen);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      bufToBePooled = workspaceBuf;
    }

    if(!usingNHWC)
      customCudaValueHeadPoolNCHW((float*)bufToBePooled,(float*)v1Mean.buf,batchSize,v1Channels,nnXLen*nnYLen,maskSumBuf);
    else
      customCudaValueHeadPoolNHWC((const float*)bufToBePooled,(float*)v1Mean.buf,batchSize,nnXLen*nnYLen,v1Channels,maskSumBuf);
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    v2Mul.apply(cudaHandles,scratch,batchSize,v1Mean.buf,v2Out.buf,workspaceBuf,workspaceBytes);
    v2Bias.apply(cudaHandles,batchSize,v2Out.buf);
    v3Mul.apply(cudaHandles,scratch,batchSize,v2Out.buf,valueBuf,workspaceBuf,workspaceBytes);
    v3Bias.apply(cudaHandles,batchSize,valueBuf);

    sv3Mul.apply(cudaHandles,scratch,batchSize,v2Out.buf,scoreValueBuf,workspaceBuf,workspaceBytes);
    sv3Bias.apply(cudaHandles,batchSize,scoreValueBuf);

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint4D(string("v1"), v1Out.buf, batchSize, v1Channels, nnXLen, nnYLen, usingNHWC, usingFP16);
    CudaUtils::debugPrint2D(string("v1 pooled"), v1Mean.buf, batchSize, v1Channels, false);
    CudaUtils::debugPrint2D(string("v2"), v2Out.buf, batchSize, v1Channels, false);
    #endif

    if(!usingFP16) {
      vOwnershipConv.apply(cudaHandles,batchSize,false,v1Out2.buf,ownershipBuf,workspaceBuf,workspaceBytes);
    }
    else {
      vOwnershipConv.apply(cudaHandles,batchSize,false,v1Out2.buf,ownershipScratch.buf,workspaceBuf,workspaceBytes);
      customCudaCopyFromHalf((const half*)ownershipScratch.buf,(float*)ownershipBuf,batchSize*ownershipChannels*nnXLen*nnYLen);
      CUDA_ERR("vOwnership copy",cudaPeekAtLastError());
    }

  }

};

//------------------------------------------------------------------------------

struct Model {
  const string name;
  const int modelVersion;
  const int maxBatchSize;
  const int nnXLen;
  const int nnYLen;
  const int numInputChannels;
  const int numInputGlobalChannels;
  const int numInputMetaChannels;
  const int numPolicyChannels;
  const int numValueChannels;
  const int numScoreValueChannels;
  const int numOwnershipChannels;
  const bool usingFP16;
  const bool usingNHWC;
  const bool inputsUsingNHWC;

  std::unique_ptr<Trunk> trunk;
  std::unique_ptr<PolicyHead> policyHead;
  std::unique_ptr<ValueHead> valueHead;
  std::unique_ptr<CudnnManager> manager;

  Model() = delete;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  Model(
    CudaHandles* cudaHandles,
    const ModelDesc* desc,
    int maxBatchSz,
    int nnX,
    int nnY,
    bool inputsUseNHWC,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    modelVersion(desc->modelVersion),
    maxBatchSize(maxBatchSz),
    nnXLen(nnX),
    nnYLen(nnY),
    numInputChannels(desc->numInputChannels),
    numInputGlobalChannels(desc->numInputGlobalChannels),
    numInputMetaChannels(desc->numInputMetaChannels),
    numPolicyChannels(desc->numPolicyChannels),
    numValueChannels(desc->numValueChannels),
    numScoreValueChannels(desc->numScoreValueChannels),
    numOwnershipChannels(desc->numOwnershipChannels),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    inputsUsingNHWC(inputsUseNHWC)
  {
    if(nnXLen > NNPos::MAX_BOARD_LEN)
      throw StringError(Global::strprintf("nnXLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)",
        nnXLen, NNPos::MAX_BOARD_LEN
      ));
    if(nnYLen > NNPos::MAX_BOARD_LEN)
      throw StringError(Global::strprintf("nnYLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)",
        nnYLen, NNPos::MAX_BOARD_LEN
      ));

    int numFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
    if(numInputChannels != numFeatures)
      throw StringError(Global::strprintf("Neural net numInputChannels (%d) was not the expected number based on version (%d)",
        numInputChannels, numFeatures
      ));
    int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
    if(numInputGlobalChannels != numGlobalFeatures)
      throw StringError(Global::strprintf("Neural net numInputGlobalChannels (%d) was not the expected number based on version (%d)",
        numInputGlobalChannels, numGlobalFeatures
      ));
    if(numInputMetaChannels > 0) {
      if(numInputMetaChannels != SGFMetadata::METADATA_INPUT_NUM_CHANNELS)
        throw StringError(Global::strprintf("Neural net numInputMetaChannels (%d) was not the expected number (%d)",
          numInputMetaChannels, SGFMetadata::METADATA_INPUT_NUM_CHANNELS
        ));
    }

    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numInputChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numInputGlobalChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numInputMetaChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numPolicyChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numValueChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numScoreValueChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numOwnershipChannels);

    manager = std::make_unique<CudnnManager>(name, maxBatchSize, nnXLen, nnYLen);
    trunk = std::make_unique<Trunk>(cudaHandles,manager.get(),&desc->trunk,nnXLen,nnYLen,inputsUseNHWC,useFP16,useNHWC);
    policyHead = std::make_unique<PolicyHead>(cudaHandles,manager.get(),&desc->policyHead,nnXLen,nnYLen,useFP16,useNHWC);
    valueHead = std::make_unique<ValueHead>(cudaHandles,manager.get(),&desc->valueHead,nnXLen,nnYLen,useFP16,useNHWC);
  }

  ~Model()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    b = trunk->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = policyHead->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = valueHead->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);

    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    bool requireExactNNLen,

    void* inputBuf,
    void* inputGlobalBuf,
    void* inputMetaBuf,

    float* policyPassBuf,
    float* policyBuf,

    float* valueBuf,
    float* scoreValueBuf,
    void* ownershipBuf,

    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> mask(scratch->allocator, scratch->getBufSizeXY(1));
    SizedBuf<void*> maskFloat(scratch->allocator, scratch->getBufSizeXYFloat(1));
    SizedBuf<void*> maskSum(scratch->allocator, scratch->getBufSizeFloat(1));

    void* maskBuf = mask.buf;
    float* maskFloatBuf = (float*)maskFloat.buf;
    float* maskSumBuf = (float*)maskSum.buf;

    if(!usingFP16) {
      if(inputsUsingNHWC)
        customCudaChannel0ExtractNHWC((const float*)inputBuf, (float*)maskBuf, batchSize, nnXLen*nnYLen, numInputChannels);
      else
        customCudaChannel0ExtractNCHW((const float*)inputBuf, (float*)maskBuf, batchSize, numInputChannels, nnXLen*nnYLen);
      CUDA_ERR("modelExtractMask",cudaPeekAtLastError());
    }
    else {
      if(inputsUsingNHWC)
        customCudaChannel0ExtractNHWC((const half*)inputBuf, (half*)maskBuf, batchSize, nnXLen*nnYLen, numInputChannels);
      else
        customCudaChannel0ExtractNCHW((const half*)inputBuf, (half*)maskBuf, batchSize, numInputChannels, nnXLen*nnYLen);
      CUDA_ERR("modelExtractMask",cudaPeekAtLastError());
    }

    fillMaskFloatBufAndMaskSumBuf(maskBuf,maskFloatBuf,maskSumBuf,usingFP16,batchSize,nnXLen,nnYLen);

    //Don't do any masking if we know the board is exactly the desired size
    if(requireExactNNLen) {
      //Set to NULL to signal downstream that this buf doesn't need to be used
      maskBuf = NULL;
      maskFloatBuf = NULL;
      //The global pooling structures need this no matter what, for normalizing based on this and its sqrt.
      //maskSumBuf = NULL;
    }

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint4D(string("Initial bin features"), inputBuf, batchSize, trunk->initialConv->inChannels, nnXLen, nnYLen, inputsUsingNHWC, usingFP16);
    CudaUtils::debugPrint2D(string("Initial global features"), inputGlobalBuf, batchSize, trunk->initialMatMul->inChannels, usingFP16);
    if(trunk->sgfMetadataEncoder != nullptr) {
      assert(inputMetaBuf != NULL);
      CudaUtils::debugPrint2D(string("Initial meta features"), inputMetaBuf, batchSize, trunk->sgfMetadataEncoder->mul1.inChannels, usingFP16);
    }
    #endif

    SizedBuf<void*> trunkBuf(scratch->allocator, scratch->getBufSizeXY(trunk->trunkNumChannels));

    trunk->apply(
      cudaHandles,
      scratch,
      batchSize,
      inputBuf,
      inputGlobalBuf,
      inputMetaBuf,
      maskBuf,
      maskSumBuf,
      trunkBuf.buf,
      workspaceBuf,
      workspaceBytes
    );
    policyHead->apply(
      cudaHandles,
      scratch,
      batchSize,
      maskBuf,
      maskFloatBuf,
      maskSumBuf,
      trunkBuf.buf,
      policyPassBuf,
      policyBuf,
      workspaceBuf,
      workspaceBytes
    );
    valueHead->apply(
      cudaHandles,
      scratch,
      batchSize,
      maskBuf,
      maskSumBuf,
      trunkBuf.buf,
      valueBuf,
      scoreValueBuf,
      ownershipBuf,
      workspaceBuf,
      workspaceBytes
    );
  }

};


//------------------------------------------------------------------------------

enum class TransformerPrecision {
  Float32,
  Float16,
  BFloat16,
};

static const char* transformerPrecisionName(TransformerPrecision precision) {
  switch(precision) {
  case TransformerPrecision::Float32:
    return "fp32";
  case TransformerPrecision::Float16:
    return "fp16";
  case TransformerPrecision::BFloat16:
    return "bf16";
  default:
    break;
  }
  ASSERT_UNREACHABLE;
  return "unknown";
}

static size_t transformerPrecisionElemSize(TransformerPrecision precision) {
  switch(precision) {
  case TransformerPrecision::Float32:
    return sizeof(float);
  case TransformerPrecision::Float16:
    return sizeof(half);
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
  case TransformerPrecision::BFloat16:
    return sizeof(bfloat16_t);
#endif
  default:
    break;
  }
  ASSERT_UNREACHABLE;
  return sizeof(float);
}

static bool transformerPrecisionUsesLowpStorage(TransformerPrecision precision) {
  return precision != TransformerPrecision::Float32;
}

static cudaDataType_t transformerPrecisionToCudaType(TransformerPrecision precision) {
  switch(precision) {
  case TransformerPrecision::Float32:
    return CUDA_R_32F;
  case TransformerPrecision::Float16:
    return CUDA_R_16F;
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
  case TransformerPrecision::BFloat16:
    return CUDA_R_16BF;
#endif
  default:
    break;
  }
  ASSERT_UNREACHABLE;
  return CUDA_R_32F;
}

static cudnnDataType_t transformerPrecisionToCudnnType(TransformerPrecision precision) {
  switch(precision) {
  case TransformerPrecision::Float32:
    return CUDNN_DATA_FLOAT;
  case TransformerPrecision::Float16:
    return CUDNN_DATA_HALF;
#ifdef KATAGO_CUDA_TRANSFORMER_BFLOAT16_AVAILABLE
  case TransformerPrecision::BFloat16:
    return CUDNN_DATA_BFLOAT16;
#endif
  default:
    break;
  }
  ASSERT_UNREACHABLE;
  return CUDNN_DATA_FLOAT;
}

enum class TransformerAttentionPath {
  Legacy,
  CudnnSdpa,
};

static const char* transformerAttentionPathName(TransformerAttentionPath path) {
  switch(path) {
  case TransformerAttentionPath::Legacy:
    return "legacy";
  case TransformerAttentionPath::CudnnSdpa:
    return "cudnn_sdpa";
  default:
    break;
  }
  ASSERT_UNREACHABLE;
  return "legacy";
}

#ifdef KATAGO_CUDNN_SDPA_AVAILABLE
static fe::DataType_t transformerPrecisionToFrontendType(TransformerPrecision precision) {
  switch(precision) {
  case TransformerPrecision::Float16:
    return fe::DataType_t::HALF;
  case TransformerPrecision::BFloat16:
    return fe::DataType_t::BFLOAT16;
  default:
    break;
  }
  ASSERT_UNREACHABLE;
  return fe::DataType_t::HALF;
}
#endif

static bool cudaDeviceSupportsTransformerFloat16(int majorComputeCapability, int minorComputeCapability) {
  return majorComputeCapability > 5 || (majorComputeCapability == 5 && minorComputeCapability >= 3);
}

static bool cudaDeviceSupportsTransformerBFloat16(int majorComputeCapability, int minorComputeCapability) {
  (void)minorComputeCapability;
#ifdef KATAGO_CUDA_TRANSFORMER_BFLOAT16_AVAILABLE
  return majorComputeCapability >= 8;
#else
  return false;
#endif
}

static bool cudaDeviceSupportsOfficialTransformerAttention(
  const CudaHandles* cudaHandles,
  TransformerPrecision precision,
  int headDim
) {
#ifdef KATAGO_CUDNN_SDPA_AVAILABLE
  if(!transformerPrecisionUsesLowpStorage(precision))
    return false;
  if(cudaHandles->majorComputeCapability < 8)
    return false;
  if(headDim <= 0 || headDim > 256 || headDim % 8 != 0)
    return false;
  if(cudnnGetVersion() < 8903)
    return false;
  return true;
#else
  (void)cudaHandles;
  (void)precision;
  (void)headDim;
  return false;
#endif
}

static TransformerPrecision chooseTransformerPrecision(
  compute_precision_t requestedMode,
  int majorComputeCapability,
  int minorComputeCapability
) {
  if(requestedMode == compute_precision_t::FP32) {
    return TransformerPrecision::Float32;
  }
  if(requestedMode == compute_precision_t::FP16) {
    if(!cudaDeviceSupportsTransformerFloat16(majorComputeCapability, minorComputeCapability))
      throw StringError("Transformer CUDA backend fp16 需要 CUDA compute capability >= 5.3");
    return TransformerPrecision::Float16;
  }
  if(requestedMode == compute_precision_t::BF16) {
    if(!cudaDeviceSupportsTransformerBFloat16(majorComputeCapability, minorComputeCapability))
      throw StringError("Transformer CUDA backend bf16 需要 CUDA compute capability >= 8.0");
    return TransformerPrecision::BFloat16;
  }
  if(requestedMode == compute_precision_t::Auto) {
    if(cudaDeviceSupportsTransformerBFloat16(majorComputeCapability, minorComputeCapability))
      return TransformerPrecision::BFloat16;
    if(cudaDeviceSupportsTransformerFloat16(majorComputeCapability, minorComputeCapability))
      return TransformerPrecision::Float16;
    return TransformerPrecision::Float32;
  }
  ASSERT_UNREACHABLE;
  return TransformerPrecision::Float32;
}

static void transformerCopyFloatToPrecision(
  const float* src,
  void* dst,
  int numElts,
  TransformerPrecision precision
) {
  switch(precision) {
  case TransformerPrecision::Float32:
    CUDA_ERR("transformerCopyFloatToPrecision", cudaMemcpy(dst, src, (size_t)numElts * sizeof(float), cudaMemcpyDeviceToDevice));
    return;
  case TransformerPrecision::Float16:
    customCudaCopyToHalf(src, reinterpret_cast<half*>(dst), numElts);
    return;
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
  case TransformerPrecision::BFloat16:
    customCudaCopyToBFloat16(src, reinterpret_cast<bfloat16_t*>(dst), numElts);
    return;
#endif
  default:
    break;
  }
  ASSERT_UNREACHABLE;
}

static void transformerCopyPrecisionToFloat(
  const void* src,
  float* dst,
  int numElts,
  TransformerPrecision precision
) {
  switch(precision) {
  case TransformerPrecision::Float32:
    CUDA_ERR("transformerCopyPrecisionToFloat", cudaMemcpy(dst, src, (size_t)numElts * sizeof(float), cudaMemcpyDeviceToDevice));
    return;
  case TransformerPrecision::Float16:
    customCudaCopyFromHalf(reinterpret_cast<const half*>(src), dst, numElts);
    return;
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
  case TransformerPrecision::BFloat16:
    customCudaCopyFromBFloat16(reinterpret_cast<const bfloat16_t*>(src), dst, numElts);
    return;
#endif
  default:
    break;
  }
  ASSERT_UNREACHABLE;
}

static void transformerMallocAndCopyWeights(
  const string& name,
  const vector<float>& weights,
  void*& deviceBuf,
  TransformerPrecision precision
) {
  if(precision == TransformerPrecision::Float32) {
    CudaUtils::mallocAndCopyToDevice(name, weights, deviceBuf, false);
    return;
  }
  if(precision == TransformerPrecision::Float16) {
    CudaUtils::mallocAndCopyToDevice(name, weights, deviceBuf, true);
    return;
  }
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
  vector<bfloat16_t> weightsBFloat(weights.size());
  for(size_t i = 0; i < weights.size(); i++)
    weightsBFloat[i] = __float2bfloat16(weights[i]);
  CUDA_ERR(name.c_str(), cudaMalloc(&deviceBuf, weightsBFloat.size() * sizeof(bfloat16_t)));
  CUDA_ERR(name.c_str(), cudaMemcpy(deviceBuf, weightsBFloat.data(), weightsBFloat.size() * sizeof(bfloat16_t), cudaMemcpyHostToDevice));
  return;
#else
  (void)name;
  (void)weights;
  (void)deviceBuf;
  throw StringError("当前 CUDA 编译环境不支持 Transformer bf16 权重");
#endif
}

static vector<float> transformerConcatLinearWeightsByOutput(
  const vector<const vector<float>*>& weightSets,
  int inChannels,
  const vector<int>& outChannelsPerSet
) {
  ASSERT(weightSets.size() == outChannelsPerSet.size());
  int totalOutChannels = 0;
  for(size_t i = 0; i < weightSets.size(); i++) {
    ASSERT(weightSets[i] != nullptr);
    ASSERT((int)weightSets[i]->size() == inChannels * outChannelsPerSet[i]);
    totalOutChannels += outChannelsPerSet[i];
  }

  vector<float> combined((size_t)inChannels * totalOutChannels);
  for(int in = 0; in < inChannels; in++) {
    int outOffset = 0;
    for(size_t i = 0; i < weightSets.size(); i++) {
      int outChannels = outChannelsPerSet[i];
      const float* src = weightSets[i]->data() + (size_t)in * outChannels;
      float* dst = combined.data() + (size_t)in * totalOutChannels + outOffset;
      std::copy(src, src + outChannels, dst);
      outOffset += outChannels;
    }
  }
  return combined;
}

static void transformerSplitCombinedLinearOutput(
  const string& name,
  const void* srcBuf,
  int rows,
  int channelsPerSplit,
  int numSplits,
  TransformerPrecision precision,
  void* const* dstBufs
) {
  size_t elemSize = transformerPrecisionElemSize(precision);
  size_t splitBytes = (size_t)channelsPerSplit * elemSize;
  size_t srcPitch = splitBytes * numSplits;
  size_t dstPitch = splitBytes;
  const char* src = reinterpret_cast<const char*>(srcBuf);
  for(int i = 0; i < numSplits; i++) {
    CUDA_ERR(name.c_str(), cudaMemcpy2DAsync(
      dstBufs[i],
      dstPitch,
      src + (size_t)i * splitBytes,
      srcPitch,
      splitBytes,
      rows,
      cudaMemcpyDeviceToDevice,
      0
    ));
  }
}

#ifdef KATAGO_CUBLASLT_AVAILABLE
static constexpr size_t TRANSFORMER_CUBLASLT_WORKSPACE_BYTES = 1 << 22;
#endif

#ifdef KATAGO_CUDNN_SDPA_AVAILABLE
struct TransformerSdpaPlan {
  std::shared_ptr<fe::graph::Graph> graph;
  int64_t workspaceBytes;

  TransformerSdpaPlan()
    : graph(), workspaceBytes(0)
  {}
};

static TransformerSdpaPlan buildTransformerSdpaPlan(
  CudaHandles* cudaHandles,
  TransformerPrecision precision,
  int batchSize,
  int seqLen,
  int numHeads,
  int headDim
) {
  const int64_t b = batchSize;
  const int64_t h = numHeads;
  const int64_t s = seqLen;
  const int64_t d = headDim;
  const float scale = 1.0f / sqrtf((float)headDim);
  fe::DataType_t ioType = transformerPrecisionToFrontendType(precision);

  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_io_data_type(ioType).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

  auto tensorDims = std::vector<int64_t>{b, h, s, d};
  auto tensorStrides = std::vector<int64_t>{h * s * d, s * d, d, 1};

  auto q = graph->tensor(
    fe::graph::Tensor_attributes()
      .set_name("Q")
      .set_dim(tensorDims)
      .set_stride(tensorStrides)
      .set_uid(1)
  );
  auto k = graph->tensor(
    fe::graph::Tensor_attributes()
      .set_name("K")
      .set_dim(tensorDims)
      .set_stride(tensorStrides)
      .set_uid(2)
  );
  auto v = graph->tensor(
    fe::graph::Tensor_attributes()
      .set_name("V")
      .set_dim(tensorDims)
      .set_stride(tensorStrides)
      .set_uid(3)
  );

  auto o = graph->sdpa(
    q,
    k,
    v,
    fe::graph::SDPA_attributes()
      .set_name("TransformerSDPA")
      .set_generate_stats(false)
      .set_attn_scale(scale)
  );

  o->set_output(true)
    .set_name("O")
    .set_uid(4)
    .set_dim(tensorDims)
    .set_stride(tensorStrides);

  auto status = graph->build(cudaHandles->cudnn, {fe::HeurMode_t::A});
  if(!status.is_good())
    throw StringError("构建 cuDNN frontend SDPA plan 失败");

  int64_t workspaceBytes = 0;
  auto workspaceStatus = graph->get_workspace_size(workspaceBytes);
  if(!workspaceStatus.is_good())
    throw StringError("获取 cuDNN frontend SDPA workspace 失败");

  TransformerSdpaPlan plan;
  plan.graph = graph;
  plan.workspaceBytes = workspaceBytes;
  return plan;
}
#endif

struct TransformerRMSNorm {
  const string name;
  const int channels;
  const TransformerPrecision precision;
  float* weightBuf;

  TransformerRMSNorm(const string& name_, const vector<float>& weights, TransformerPrecision precision_)
    : name(name_), channels((int)weights.size()), precision(precision_), weightBuf(nullptr)
  {
    void* buf = nullptr;
    CudaUtils::mallocAndCopyToDevice(name, weights, buf, false);
    weightBuf = reinterpret_cast<float*>(buf);
  }

  ~TransformerRMSNorm() {
    cudaFree(weightBuf);
  }

  TransformerRMSNorm() = delete;
  TransformerRMSNorm(const TransformerRMSNorm&) = delete;
  TransformerRMSNorm& operator=(const TransformerRMSNorm&) = delete;

  void apply(int rows, const void* inputBuf, void* outputBuf) const {
    if(precision == TransformerPrecision::Float32)
      customCudaRMSNorm(reinterpret_cast<const float*>(inputBuf), reinterpret_cast<float*>(outputBuf), weightBuf, rows, channels, 1e-6f);
    else if(precision == TransformerPrecision::Float16)
      customCudaRMSNorm(reinterpret_cast<const half*>(inputBuf), reinterpret_cast<half*>(outputBuf), weightBuf, rows, channels, 1e-6f);
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
    else if(precision == TransformerPrecision::BFloat16)
      customCudaRMSNorm(reinterpret_cast<const bfloat16_t*>(inputBuf), reinterpret_cast<bfloat16_t*>(outputBuf), weightBuf, rows, channels, 1e-6f);
#endif
    else
      ASSERT_UNREACHABLE;
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());
  }
};

struct TransformerStemConv {
  const string name;
  const int inChannels;
  const int outChannels;
  const int kernelSize;
  const int nnXLen;
  const int nnYLen;
  const bool inputUsingNHWC;
  const TransformerPrecision precision;

  ByBatchSize<cudnnTensorDescriptor_t>* inputDescriptors;
  ByBatchSize<cudnnTensorDescriptor_t>* outputDescriptors;
#if CUDNN_MAJOR >= 8
  ByBatchSize<cudnnConvolutionFwdAlgoPerf_t>* convolutionAlgorithms;
#else
  ByBatchSize<cudnnConvolutionFwdAlgo_t>* convolutionAlgorithms;
#endif
  cudnnFilterDescriptor_t filterDescriptor;
  cudnnConvolutionDescriptor_t convolutionDescriptor;
  void* filterBuf;

  TransformerStemConv() = delete;
  TransformerStemConv(const TransformerStemConv&) = delete;
  TransformerStemConv& operator=(const TransformerStemConv&) = delete;

  TransformerStemConv(
    CudaHandles* cudaHandles,
    const string& name_,
    int maxBatchSize,
    int nnXLen_,
    int nnYLen_,
    bool inputUsingNHWC_,
    int inChannels_,
    int outChannels_,
    int kernelSize_,
    const vector<float>& weights,
    TransformerPrecision precision_
  ) :
    name(name_),
    inChannels(inChannels_),
    outChannels(outChannels_),
    kernelSize(kernelSize_),
    nnXLen(nnXLen_),
    nnYLen(nnYLen_),
    inputUsingNHWC(inputUsingNHWC_),
    precision(precision_),
    inputDescriptors(new ByBatchSize<cudnnTensorDescriptor_t>(maxBatchSize)),
    outputDescriptors(new ByBatchSize<cudnnTensorDescriptor_t>(maxBatchSize)),
#if CUDNN_MAJOR >= 8
    convolutionAlgorithms(new ByBatchSize<cudnnConvolutionFwdAlgoPerf_t>(maxBatchSize)),
#else
    convolutionAlgorithms(new ByBatchSize<cudnnConvolutionFwdAlgo_t>(maxBatchSize)),
#endif
    filterBuf(nullptr)
  {
    cudnnDataType_t tensorType = transformerPrecisionToCudnnType(precision);
    bool tensorCoresSupported = cudaHandles->majorComputeCapability >= 7;

    inputDescriptors->destroyFunc = cudnnDestroyTensorDescriptor;
    outputDescriptors->destroyFunc = cudnnDestroyTensorDescriptor;
    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      cudnnTensorDescriptor_t& inputDesc = (*inputDescriptors)[batchSize];
      cudnnTensorDescriptor_t& outputDesc = (*outputDescriptors)[batchSize];
      CUDNN_ERR(name.c_str(), cudnnCreateTensorDescriptor(&inputDesc));
      CUDNN_ERR(name.c_str(), cudnnSetTensor4dDescriptor(
        inputDesc,
        inputUsingNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
        tensorType,
        batchSize,
        inChannels,
        nnYLen,
        nnXLen
      ));
      CUDNN_ERR(name.c_str(), cudnnCreateTensorDescriptor(&outputDesc));
      CUDNN_ERR(name.c_str(), cudnnSetTensor4dDescriptor(
        outputDesc,
        CUDNN_TENSOR_NCHW,
        tensorType,
        batchSize,
        outChannels,
        nnYLen,
        nnXLen
      ));
    }

    CUDNN_ERR(name.c_str(), cudnnCreateFilterDescriptor(&filterDescriptor));
    CUDNN_ERR(name.c_str(), cudnnSetFilter4dDescriptor(
      filterDescriptor,
      tensorType,
      CUDNN_TENSOR_NCHW,
      outChannels,
      inChannels,
      kernelSize,
      kernelSize
    ));

    CUDNN_ERR(name.c_str(), cudnnCreateConvolutionDescriptor(&convolutionDescriptor));
    CUDNN_ERR(name.c_str(), cudnnSetConvolution2dDescriptor(
      convolutionDescriptor,
      kernelSize / 2,
      kernelSize / 2,
      1,
      1,
      1,
      1,
      CUDNN_CROSS_CORRELATION,
      CUDNN_DATA_FLOAT
    ));
    if(transformerPrecisionUsesLowpStorage(precision) && tensorCoresSupported)
      CUDNN_ERR(name.c_str(), cudnnSetConvolutionMathType(convolutionDescriptor, CUDNN_TENSOR_OP_MATH));

    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      if(transformerPrecisionUsesLowpStorage(precision)) {
#if CUDNN_MAJOR >= 8
        (*convolutionAlgorithms)[batchSize].algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
#else
        (*convolutionAlgorithms)[batchSize] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
#endif
      }
      else {
#if CUDNN_MAJOR >= 8
        int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
        int returnedAlgoCount = -1;
        cudnnConvolutionFwdAlgoPerf_t results[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
        CUDNN_ERR(name.c_str(), cudnnGetConvolutionForwardAlgorithm_v7(
          cudaHandles->cudnn,
          (*inputDescriptors)[batchSize],
          filterDescriptor,
          convolutionDescriptor,
          (*outputDescriptors)[batchSize],
          requestedAlgoCount,
          &returnedAlgoCount,
          results
        ));
        if(returnedAlgoCount <= 0)
          throw StringError("Transformer stem cudnnGetConvolutionForwardAlgorithm_v7 returned no algorithms");
        (*convolutionAlgorithms)[batchSize] = results[0];
#else
        size_t bytesMemoryLimit = 0;
        CUDNN_ERR(name.c_str(), cudnnGetConvolutionForwardAlgorithm(
          cudaHandles->cudnn,
          (*inputDescriptors)[batchSize],
          filterDescriptor,
          convolutionDescriptor,
          (*outputDescriptors)[batchSize],
          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
          bytesMemoryLimit,
          &((*convolutionAlgorithms)[batchSize])
        ));
#endif
      }
    }

    assert(weights.size() == (size_t)outChannels * inChannels * kernelSize * kernelSize);
    transformerMallocAndCopyWeights(name, weights, filterBuf, precision);
  }

  ~TransformerStemConv() {
    delete inputDescriptors;
    delete outputDescriptors;
    delete convolutionAlgorithms;
    cudaFree(filterBuf);
    cudnnDestroyFilterDescriptor(filterDescriptor);
    cudnnDestroyConvolutionDescriptor(convolutionDescriptor);
  }

  size_t requiredWorkspaceBytes(CudaHandles* cudaHandles, int batchSize) const {
    size_t workspaceBytes = 0;
#if CUDNN_MAJOR >= 8
    CUDNN_ERR(name.c_str(), cudnnGetConvolutionForwardWorkspaceSize(
      cudaHandles->cudnn,
      (*inputDescriptors)[batchSize],
      filterDescriptor,
      convolutionDescriptor,
      (*outputDescriptors)[batchSize],
      (*convolutionAlgorithms)[batchSize].algo,
      &workspaceBytes
    ));
#else
    CUDNN_ERR(name.c_str(), cudnnGetConvolutionForwardWorkspaceSize(
      cudaHandles->cudnn,
      (*inputDescriptors)[batchSize],
      filterDescriptor,
      convolutionDescriptor,
      (*outputDescriptors)[batchSize],
      (*convolutionAlgorithms)[batchSize],
      &workspaceBytes
    ));
#endif
    return workspaceBytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    const float* inputBuf,
    float* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    if(precision == TransformerPrecision::Float32) {
      const float alpha = 1.0f;
      const float beta = 0.0f;
#if CUDNN_MAJOR >= 8
      CUDNN_ERR(name.c_str(), cudnnConvolutionForward(
        cudaHandles->cudnn,
        &alpha,
        (*inputDescriptors)[batchSize],
        inputBuf,
        filterDescriptor,
        filterBuf,
        convolutionDescriptor,
        (*convolutionAlgorithms)[batchSize].algo,
        workspaceBuf,
        workspaceBytes,
        &beta,
        (*outputDescriptors)[batchSize],
        outputBuf
      ));
#else
      CUDNN_ERR(name.c_str(), cudnnConvolutionForward(
        cudaHandles->cudnn,
        &alpha,
        (*inputDescriptors)[batchSize],
        inputBuf,
        filterDescriptor,
        filterBuf,
        convolutionDescriptor,
        (*convolutionAlgorithms)[batchSize],
        workspaceBuf,
        workspaceBytes,
        &beta,
        (*outputDescriptors)[batchSize],
        outputBuf
      ));
#endif
      return;
    }

    int numInputElts = batchSize * inChannels * nnXLen * nnYLen;
    int numOutputElts = batchSize * outChannels * nnXLen * nnYLen;
    SizedBuf<void*> lowInputBuf(scratch->allocator, (size_t)numInputElts * transformerPrecisionElemSize(precision));
    SizedBuf<void*> lowOutputBuf(scratch->allocator, (size_t)numOutputElts * transformerPrecisionElemSize(precision));
    transformerCopyFloatToPrecision(inputBuf, lowInputBuf.buf, numInputElts, precision);
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());

    const float alpha = 1.0f;
    const float beta = 0.0f;
#if CUDNN_MAJOR >= 8
    CUDNN_ERR(name.c_str(), cudnnConvolutionForward(
      cudaHandles->cudnn,
      &alpha,
      (*inputDescriptors)[batchSize],
      lowInputBuf.buf,
      filterDescriptor,
      filterBuf,
      convolutionDescriptor,
      (*convolutionAlgorithms)[batchSize].algo,
      workspaceBuf,
      workspaceBytes,
      &beta,
      (*outputDescriptors)[batchSize],
      lowOutputBuf.buf
    ));
#else
    CUDNN_ERR(name.c_str(), cudnnConvolutionForward(
      cudaHandles->cudnn,
      &alpha,
      (*inputDescriptors)[batchSize],
      lowInputBuf.buf,
      filterDescriptor,
      filterBuf,
      convolutionDescriptor,
      (*convolutionAlgorithms)[batchSize],
      workspaceBuf,
      workspaceBytes,
      &beta,
      (*outputDescriptors)[batchSize],
      lowOutputBuf.buf
    ));
#endif

    transformerCopyPrecisionToFloat(lowOutputBuf.buf, outputBuf, numOutputElts, precision);
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());
  }
};

struct TransformerLinear {
  const string name;
  const int inChannels;
  const int outChannels;
  const TransformerPrecision precision;
  void* weightBuf;

  TransformerLinear(
    CudaHandles* cudaHandles,
    const string& name_,
    int inChannels_,
    int outChannels_,
    const vector<float>& weights,
    TransformerPrecision precision_
  ) :
    name(name_),
    inChannels(inChannels_),
    outChannels(outChannels_),
    precision(precision_),
    weightBuf(nullptr)
  {
    (void)cudaHandles;
    transformerMallocAndCopyWeights(name_, weights, weightBuf, precision);
  }

  ~TransformerLinear() {
    cudaFree(weightBuf);
  }

  TransformerLinear() = delete;
  TransformerLinear(const TransformerLinear&) = delete;
  TransformerLinear& operator=(const TransformerLinear&) = delete;

  void runFloatMatmul(
    CudaHandles* cudaHandles,
    int batchSize,
    const float* inputBuf,
    float* outputBuf
  ) const {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_ERR(name.c_str(), cublasSgemm(
      cudaHandles->cublas,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      outChannels,
      batchSize,
      inChannels,
      &alpha,
      reinterpret_cast<const float*>(weightBuf), outChannels,
      inputBuf, inChannels,
      &beta,
      outputBuf, outChannels
    ));
  }

#ifdef KATAGO_CUBLASLT_AVAILABLE
  bool tryRunLowpMatmulWithCublasLt(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    const void* lowpInputBuf,
    void* lowpOutputBuf
  ) const {
    ASSERT(transformerPrecisionUsesLowpStorage(precision));
    cudaDataType_t dtype = transformerPrecisionToCudaType(precision);

    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t weightLayout = nullptr;
    cublasLtMatrixLayout_t inputLayout = nullptr;
    cublasLtMatrixLayout_t outputLayout = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    size_t maxWorkspaceBytes = TRANSFORMER_CUBLASLT_WORKSPACE_BYTES;
    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedResults = 0;
    bool success = false;

    CUBLAS_ERR(name.c_str(), cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_ERR(name.c_str(), cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CUBLAS_ERR(name.c_str(), cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    CUBLAS_ERR(name.c_str(), cublasLtMatrixLayoutCreate(&weightLayout, dtype, outChannels, inChannels, outChannels));
    CUBLAS_ERR(name.c_str(), cublasLtMatrixLayoutCreate(&inputLayout, dtype, inChannels, batchSize, inChannels));
    CUBLAS_ERR(name.c_str(), cublasLtMatrixLayoutCreate(&outputLayout, dtype, outChannels, batchSize, outChannels));
    CUBLAS_ERR(name.c_str(), cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_ERR(name.c_str(), cublasLtMatmulPreferenceSetAttribute(
      preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &maxWorkspaceBytes,
      sizeof(maxWorkspaceBytes)
    ));

    cublasStatus_t heuristicStatus = cublasLtMatmulAlgoGetHeuristic(
      cudaHandles->cublasLt,
      operationDesc,
      weightLayout,
      inputLayout,
      outputLayout,
      outputLayout,
      preference,
      1,
      &heuristicResult,
      &returnedResults
    );
    if(heuristicStatus == CUBLAS_STATUS_SUCCESS && returnedResults > 0) {
      void* workspacePtr = nullptr;
      if(heuristicResult.workspaceSize > 0) {
        SizedBuf<void*> workspaceBuf(scratch->allocator, heuristicResult.workspaceSize);
        workspacePtr = workspaceBuf.buf;
        CUBLAS_ERR(name.c_str(), cublasLtMatmul(
          cudaHandles->cublasLt,
          operationDesc,
          &alpha,
          weightBuf,
          weightLayout,
          lowpInputBuf,
          inputLayout,
          &beta,
          lowpOutputBuf,
          outputLayout,
          lowpOutputBuf,
          outputLayout,
          &heuristicResult.algo,
          workspacePtr,
          heuristicResult.workspaceSize,
          0
        ));
      }
      else {
        CUBLAS_ERR(name.c_str(), cublasLtMatmul(
          cudaHandles->cublasLt,
          operationDesc,
          &alpha,
          weightBuf,
          weightLayout,
          lowpInputBuf,
          inputLayout,
          &beta,
          lowpOutputBuf,
          outputLayout,
          lowpOutputBuf,
          outputLayout,
          &heuristicResult.algo,
          nullptr,
          0,
          0
        ));
      }
      success = true;
    }

    if(preference != nullptr)
      cublasLtMatmulPreferenceDestroy(preference);
    if(outputLayout != nullptr)
      cublasLtMatrixLayoutDestroy(outputLayout);
    if(inputLayout != nullptr)
      cublasLtMatrixLayoutDestroy(inputLayout);
    if(weightLayout != nullptr)
      cublasLtMatrixLayoutDestroy(weightLayout);
    if(operationDesc != nullptr)
      cublasLtMatmulDescDestroy(operationDesc);

    return success;
  }
#endif

  void runLowpMatmul(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    const void* lowpInputBuf,
    void* lowpOutputBuf
  ) const {
    ASSERT(transformerPrecisionUsesLowpStorage(precision));
#ifdef KATAGO_CUBLASLT_AVAILABLE
    if(tryRunLowpMatmulWithCublasLt(cudaHandles, scratch, batchSize, lowpInputBuf, lowpOutputBuf))
      return;
#else
    (void)scratch;
#endif

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudaDataType_t dtype = transformerPrecisionToCudaType(precision);
    CUBLAS_ERR(name.c_str(), cublasGemmEx(
      cudaHandles->cublas,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      outChannels,
      batchSize,
      inChannels,
      &alpha,
      weightBuf,
      dtype,
      outChannels,
      lowpInputBuf,
      dtype,
      inChannels,
      &beta,
      lowpOutputBuf,
      dtype,
      outChannels,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT
    ));
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    const void* inputBuf,
    void* outputBuf
  ) const {
    const float* inputFloatBuf = reinterpret_cast<const float*>(inputBuf);
    float* outputFloatBuf = reinterpret_cast<float*>(outputBuf);
    if(precision == TransformerPrecision::Float32) {
      runFloatMatmul(cudaHandles, batchSize, inputFloatBuf, outputFloatBuf);
      return;
    }

    int numInputElts = batchSize * inChannels;
    int numOutputElts = batchSize * outChannels;
    SizedBuf<void*> lowInputBuf(scratch->allocator, (size_t)numInputElts * transformerPrecisionElemSize(precision));
    SizedBuf<void*> lowOutputBuf(scratch->allocator, (size_t)numOutputElts * transformerPrecisionElemSize(precision));

    transformerCopyFloatToPrecision(inputFloatBuf, lowInputBuf.buf, numInputElts, precision);
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());

    runLowpMatmul(cudaHandles, scratch, batchSize, lowInputBuf.buf, lowOutputBuf.buf);
    transformerCopyPrecisionToFloat(lowOutputBuf.buf, outputFloatBuf, numOutputElts, precision);
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());
  }

  // inputBuf is already in lowp format (or FP32 if precision==Float32).
  // outputBuf is always FP32.
  void applyWithLowpInput(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    const void* lowpInputBuf,
    void* outputBuf
  ) const {
    float* outputFloatBuf = reinterpret_cast<float*>(outputBuf);
    if(precision == TransformerPrecision::Float32) {
      runFloatMatmul(cudaHandles, batchSize, reinterpret_cast<const float*>(lowpInputBuf), outputFloatBuf);
      return;
    }

    int numOutputElts = batchSize * outChannels;
    SizedBuf<void*> lowOutputBuf(scratch->allocator, (size_t)numOutputElts * transformerPrecisionElemSize(precision));

    runLowpMatmul(cudaHandles, scratch, batchSize, lowpInputBuf, lowOutputBuf.buf);
    transformerCopyPrecisionToFloat(lowOutputBuf.buf, outputFloatBuf, numOutputElts, precision);
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());
  }

  // inputBuf is float, outputBuf is lowp (or FP32 if precision==Float32).
  void applyToLowp(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    const void* inputBuf,
    void* lowpOutputBuf
  ) const {
    if(precision == TransformerPrecision::Float32) {
      runFloatMatmul(cudaHandles, batchSize, reinterpret_cast<const float*>(inputBuf), reinterpret_cast<float*>(lowpOutputBuf));
      return;
    }

    int numInputElts = batchSize * inChannels;
    SizedBuf<void*> lowInputBuf(scratch->allocator, (size_t)numInputElts * transformerPrecisionElemSize(precision));
    transformerCopyFloatToPrecision(reinterpret_cast<const float*>(inputBuf), lowInputBuf.buf, numInputElts, precision);
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    applyWithLowpInputToLowp(cudaHandles, scratch, batchSize, lowInputBuf.buf, lowpOutputBuf);
  }

  // inputBuf is already lowp format (or FP32 if precision==Float32), outputBuf matches precision.
  void applyWithLowpInputToLowp(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    const void* lowpInputBuf,
    void* lowpOutputBuf
  ) const {
    if(precision == TransformerPrecision::Float32) {
      runFloatMatmul(cudaHandles, batchSize, reinterpret_cast<const float*>(lowpInputBuf), reinterpret_cast<float*>(lowpOutputBuf));
      return;
    }

    runLowpMatmul(cudaHandles, scratch, batchSize, lowpInputBuf, lowpOutputBuf);
  }
};

struct TransformerBlock {
  const string name;
  const int hiddenSize;
  const int numHeads;
  const int headDim;
  const int ffnDim;
  const int seqLen;
  const TransformerPrecision precision;
  TransformerAttentionPath attentionPath;
  ByBatchSize<cudnnTensorDescriptor_t>* softmaxDescriptors;
#ifdef KATAGO_CUDNN_SDPA_AVAILABLE
  vector<TransformerSdpaPlan> officialAttentionPlans;
#endif

  TransformerRMSNorm norm1;
  TransformerLinear qkvProj;
  TransformerLinear outProj;
  TransformerRMSNorm norm2;
  TransformerLinear ffnInProj;
  TransformerLinear ffnW2;

  TransformerBlock() = delete;
  TransformerBlock(const TransformerBlock&) = delete;
  TransformerBlock& operator=(const TransformerBlock&) = delete;

  TransformerBlock(
    CudaHandles* cudaHandles,
    const TransformerBlockDesc& desc,
    int hiddenSize_,
    int numHeads_,
    int headDim_,
    int ffnDim_,
    int seqLen_,
    int maxBatchSize,
    int blockIdx,
    TransformerPrecision precision_
  ) :
    name("transformer_block_" + Global::intToString(blockIdx)),
    hiddenSize(hiddenSize_),
    numHeads(numHeads_),
    headDim(headDim_),
    ffnDim(ffnDim_),
    seqLen(seqLen_),
    precision(precision_),
    attentionPath(TransformerAttentionPath::Legacy),
    softmaxDescriptors(nullptr),
#ifdef KATAGO_CUDNN_SDPA_AVAILABLE
    officialAttentionPlans(),
#endif
    norm1(name + "_norm1", desc.norm1Weight, precision_),
    qkvProj(
      cudaHandles,
      name + "_qkv",
      hiddenSize,
      hiddenSize * 3,
      transformerConcatLinearWeightsByOutput(
        vector<const vector<float>*>{&desc.qWeight, &desc.kWeight, &desc.vWeight},
        hiddenSize,
        vector<int>{hiddenSize, hiddenSize, hiddenSize}
      ),
      precision_
    ),
    outProj(cudaHandles, name + "_out", hiddenSize, hiddenSize, desc.outWeight, precision_),
    norm2(name + "_norm2", desc.norm2Weight, precision_),
    ffnInProj(
      cudaHandles,
      name + "_ff_in",
      hiddenSize,
      ffnDim * 2,
      transformerConcatLinearWeightsByOutput(
        vector<const vector<float>*>{&desc.ffnW1Weight, &desc.ffnWGateWeight},
        hiddenSize,
        vector<int>{ffnDim, ffnDim}
      ),
      precision_
    ),
    ffnW2(cudaHandles, name + "_ff2", ffnDim, hiddenSize, desc.ffnW2Weight, precision_)
  {
#ifdef KATAGO_CUDNN_SDPA_AVAILABLE
    if(cudaDeviceSupportsOfficialTransformerAttention(cudaHandles, precision, headDim)) {
      try {
        officialAttentionPlans.reserve(maxBatchSize);
        for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++)
          officialAttentionPlans.push_back(buildTransformerSdpaPlan(cudaHandles, precision, batchSize, seqLen, numHeads, headDim));
        attentionPath = TransformerAttentionPath::CudnnSdpa;
      }
      catch(const std::exception&) {
        officialAttentionPlans.clear();
        attentionPath = TransformerAttentionPath::Legacy;
      }
    }
#endif

    if(attentionPath == TransformerAttentionPath::Legacy) {
      softmaxDescriptors = new ByBatchSize<cudnnTensorDescriptor_t>(maxBatchSize);
      softmaxDescriptors->destroyFunc = cudnnDestroyTensorDescriptor;
      for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
        cudnnTensorDescriptor_t& descForBatch = (*softmaxDescriptors)[batchSize];
        CUDNN_ERR(name.c_str(), cudnnCreateTensorDescriptor(&descForBatch));
        CUDNN_ERR(name.c_str(), cudnnSetTensor4dDescriptor(
          descForBatch,
          CUDNN_TENSOR_NCHW,
          CUDNN_DATA_FLOAT,
          batchSize * numHeads * seqLen,
          seqLen,
          1,
          1
        ));
      }
    }
  }

  ~TransformerBlock() {
    delete softmaxDescriptors;
  }

  bool usesOfficialAttention() const {
    return attentionPath == TransformerAttentionPath::CudnnSdpa;
  }

  const char* attentionPathName() const {
    return transformerAttentionPathName(attentionPath);
  }

  size_t requiredWorkspaceBytes(int batchSize) const {
#ifdef KATAGO_CUDNN_SDPA_AVAILABLE
    if(attentionPath == TransformerAttentionPath::CudnnSdpa) {
      ASSERT(batchSize >= 1 && batchSize <= (int)officialAttentionPlans.size());
      return (size_t)officialAttentionPlans[batchSize-1].workspaceBytes;
    }
#endif
    (void)batchSize;
    return 0;
  }

#ifdef KATAGO_CUDNN_SDPA_AVAILABLE
  void executeOfficialAttention(
    CudaHandles* cudaHandles,
    int batchSize,
    const void* qBuf,
    const void* kBuf,
    const void* vBuf,
    void* outBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    ASSERT(attentionPath == TransformerAttentionPath::CudnnSdpa);
    ASSERT(batchSize >= 1 && batchSize <= (int)officialAttentionPlans.size());
    const TransformerSdpaPlan& plan = officialAttentionPlans[batchSize-1];
    if((size_t)plan.workspaceBytes > workspaceBytes)
      throw StringError("Transformer cuDNN SDPA workspace 不足");

    std::unordered_map<int64_t, void*> variantPack = {
      {1, const_cast<void*>(qBuf)},
      {2, const_cast<void*>(kBuf)},
      {3, const_cast<void*>(vBuf)},
      {4, outBuf},
    };
    void* executeWorkspace = plan.workspaceBytes > 0 ? workspaceBuf : nullptr;
    auto status = plan.graph->execute(cudaHandles->cudnn, variantPack, executeWorkspace);
    if(!status.is_good())
      throw StringError("执行 Transformer cuDNN SDPA 失败");
  }
#endif

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    const float* ropeCosBuf,
    const float* ropeSinBuf,
    void* xBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    int tokenCount = batchSize * seqLen;
    size_t residentEltSize = transformerPrecisionElemSize(precision);
    size_t hiddenBytes = (size_t)tokenCount * hiddenSize * residentEltSize;
    size_t ffnBytes = (size_t)tokenCount * ffnDim * residentEltSize;
    size_t logitsBytes = (size_t)batchSize * numHeads * seqLen * seqLen * sizeof(float);
    size_t logitsLowpBytes = (size_t)batchSize * numHeads * seqLen * seqLen * residentEltSize;
    size_t floatHiddenBytes = (size_t)tokenCount * hiddenSize * sizeof(float);
    bool useOfficialAttention = attentionPath == TransformerAttentionPath::CudnnSdpa;
    size_t attentionResidentBytes = useOfficialAttention ? hiddenBytes : floatHiddenBytes;
    size_t qkvInterleavedBytes = hiddenBytes * 3;
    size_t ffnInterleavedBytes = ffnBytes * 2;

    SizedBuf<void*> normBuf(scratch->allocator, hiddenBytes);
    SizedBuf<void*> qkvInterleavedBuf(scratch->allocator, qkvInterleavedBytes);
    SizedBuf<void*> qBuf(scratch->allocator, hiddenBytes);
    SizedBuf<void*> kBuf(scratch->allocator, hiddenBytes);
    SizedBuf<void*> vBuf(scratch->allocator, hiddenBytes);
    SizedBuf<void*> logitsBuf(scratch->allocator, useOfficialAttention ? 1 : logitsBytes);
    SizedBuf<void*> attnLowpBuf(scratch->allocator, (!useOfficialAttention && transformerPrecisionUsesLowpStorage(precision)) ? logitsLowpBytes : 1);
    SizedBuf<void*> attnBuf(scratch->allocator, attentionResidentBytes);
    SizedBuf<void*> projBuf(scratch->allocator, hiddenBytes);
    SizedBuf<void*> norm2Buf(scratch->allocator, hiddenBytes);
    SizedBuf<void*> ffnInterleavedBuf(scratch->allocator, ffnInterleavedBytes);
    SizedBuf<void*> ffn1Buf(scratch->allocator, ffnBytes);
    SizedBuf<void*> ffnGateBuf(scratch->allocator, ffnBytes);
    SizedBuf<void*> ffnActBuf(scratch->allocator, ffnBytes);
    SizedBuf<void*> ffnOutBuf(scratch->allocator, hiddenBytes);

    // Temporary BHSD-format buffers for cuBLAS batched GEMM attention
    SizedBuf<void*> qBHSD(scratch->allocator, hiddenBytes);
    SizedBuf<void*> kBHSD(scratch->allocator, hiddenBytes);
    SizedBuf<void*> vBHSD(scratch->allocator, hiddenBytes);
    SizedBuf<void*> outBHSD(scratch->allocator, attentionResidentBytes);

    norm1.apply(tokenCount, xBuf, normBuf.buf);
    qkvProj.applyWithLowpInputToLowp(cudaHandles, scratch, tokenCount, normBuf.buf, qkvInterleavedBuf.buf);
    {
      void* qkvDstBufs[3] = {qBuf.buf, kBuf.buf, vBuf.buf};
      transformerSplitCombinedLinearOutput(name, qkvInterleavedBuf.buf, tokenCount, hiddenSize, 3, precision, qkvDstBufs);
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    }

    if(precision == TransformerPrecision::Float32)
      customCudaApplyRotaryInplace(reinterpret_cast<float*>(qBuf.buf), reinterpret_cast<float*>(kBuf.buf), ropeCosBuf, ropeSinBuf, batchSize, seqLen, numHeads, headDim);
    else if(precision == TransformerPrecision::Float16)
      customCudaApplyRotaryInplace(reinterpret_cast<half*>(qBuf.buf), reinterpret_cast<half*>(kBuf.buf), ropeCosBuf, ropeSinBuf, batchSize, seqLen, numHeads, headDim);
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
    else if(precision == TransformerPrecision::BFloat16)
      customCudaApplyRotaryInplace(reinterpret_cast<bfloat16_t*>(qBuf.buf), reinterpret_cast<bfloat16_t*>(kBuf.buf), ropeCosBuf, ropeSinBuf, batchSize, seqLen, numHeads, headDim);
#endif
    else
      ASSERT_UNREACHABLE;
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());

    // Transpose Q/K/V from [B,S,H,D] to [B,H,S,D] for cuBLAS strided batched GEMM
    if(precision == TransformerPrecision::Float32) {
      customCudaTransposeBSHDtoBHSD(reinterpret_cast<float*>(qBuf.buf), reinterpret_cast<float*>(qBHSD.buf), batchSize, seqLen, numHeads, headDim);
      customCudaTransposeBSHDtoBHSD(reinterpret_cast<float*>(kBuf.buf), reinterpret_cast<float*>(kBHSD.buf), batchSize, seqLen, numHeads, headDim);
      customCudaTransposeBSHDtoBHSD(reinterpret_cast<float*>(vBuf.buf), reinterpret_cast<float*>(vBHSD.buf), batchSize, seqLen, numHeads, headDim);
    }
    else if(precision == TransformerPrecision::Float16) {
      customCudaTransposeBSHDtoBHSD(reinterpret_cast<half*>(qBuf.buf), reinterpret_cast<half*>(qBHSD.buf), batchSize, seqLen, numHeads, headDim);
      customCudaTransposeBSHDtoBHSD(reinterpret_cast<half*>(kBuf.buf), reinterpret_cast<half*>(kBHSD.buf), batchSize, seqLen, numHeads, headDim);
      customCudaTransposeBSHDtoBHSD(reinterpret_cast<half*>(vBuf.buf), reinterpret_cast<half*>(vBHSD.buf), batchSize, seqLen, numHeads, headDim);
    }
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
    else if(precision == TransformerPrecision::BFloat16) {
      customCudaTransposeBSHDtoBHSD(reinterpret_cast<bfloat16_t*>(qBuf.buf), reinterpret_cast<bfloat16_t*>(qBHSD.buf), batchSize, seqLen, numHeads, headDim);
      customCudaTransposeBSHDtoBHSD(reinterpret_cast<bfloat16_t*>(kBuf.buf), reinterpret_cast<bfloat16_t*>(kBHSD.buf), batchSize, seqLen, numHeads, headDim);
      customCudaTransposeBSHDtoBHSD(reinterpret_cast<bfloat16_t*>(vBuf.buf), reinterpret_cast<bfloat16_t*>(vBHSD.buf), batchSize, seqLen, numHeads, headDim);
    }
#endif
    else
      ASSERT_UNREACHABLE;
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());

    if(useOfficialAttention) {
#ifdef KATAGO_CUDNN_SDPA_AVAILABLE
      executeOfficialAttention(cudaHandles, batchSize, qBHSD.buf, kBHSD.buf, vBHSD.buf, outBHSD.buf, workspaceBuf, workspaceBytes);
      if(precision == TransformerPrecision::Float16)
        customCudaTransposeBHSDtoBSHD(reinterpret_cast<half*>(outBHSD.buf), reinterpret_cast<half*>(attnBuf.buf), batchSize, seqLen, numHeads, headDim);
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
      else if(precision == TransformerPrecision::BFloat16)
        customCudaTransposeBHSDtoBSHD(reinterpret_cast<bfloat16_t*>(outBHSD.buf), reinterpret_cast<bfloat16_t*>(attnBuf.buf), batchSize, seqLen, numHeads, headDim);
#endif
      else
        ASSERT_UNREACHABLE;
#else
      ASSERT_UNREACHABLE;
#endif
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());
      outProj.applyWithLowpInputToLowp(cudaHandles, scratch, tokenCount, attnBuf.buf, projBuf.buf);
    }
    else {
      // Q @ K^T via cuBLAS strided batched GEMM
      // Row-major: logits[S,S] = Q[S,D] * K[S,D]^T * scale
      // cuBLAS col-major mapping: swap A/B, use OP_T on K
      {
        float scale = 1.0f / sqrtf((float)headDim);
        float zero = 0.0f;
        long long strideQK = (long long)seqLen * headDim;
        long long strideLogits = (long long)seqLen * seqLen;
        if(precision == TransformerPrecision::Float32) {
          CUBLAS_ERR(name.c_str(), cublasSgemmStridedBatched(
            cudaHandles->cublas,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            seqLen,
            seqLen,
            headDim,
            &scale,
            reinterpret_cast<float*>(kBHSD.buf), headDim, strideQK,
            reinterpret_cast<float*>(qBHSD.buf), headDim, strideQK,
            &zero,
            reinterpret_cast<float*>(logitsBuf.buf), seqLen, strideLogits,
            batchSize * numHeads
          ));
        }
        else {
          cudaDataType_t dtype = transformerPrecisionToCudaType(precision);
          CUBLAS_ERR(name.c_str(), cublasGemmStridedBatchedEx(
            cudaHandles->cublas,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            seqLen,
            seqLen,
            headDim,
            &scale,
            kBHSD.buf, dtype, headDim, strideQK,
            qBHSD.buf, dtype, headDim, strideQK,
            &zero,
            logitsBuf.buf, CUDA_R_32F, seqLen, strideLogits,
            batchSize * numHeads,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT
          ));
        }
      }

      // Softmax via cuDNN
      {
        float one = 1.0f;
        float zero = 0.0f;
        CUDNN_ERR(name.c_str(), cudnnSoftmaxForward(
          cudaHandles->cudnn,
          CUDNN_SOFTMAX_ACCURATE,
          CUDNN_SOFTMAX_MODE_INSTANCE,
          &one, (*softmaxDescriptors)[batchSize], logitsBuf.buf,
          &zero, (*softmaxDescriptors)[batchSize], logitsBuf.buf));
      }

      if(transformerPrecisionUsesLowpStorage(precision)) {
        int numLogitsElts = batchSize * numHeads * seqLen * seqLen;
        transformerCopyFloatToPrecision(reinterpret_cast<const float*>(logitsBuf.buf), attnLowpBuf.buf, numLogitsElts, precision);
        CUDA_ERR(name.c_str(), cudaPeekAtLastError());
      }

      // Attn @ V via cuBLAS strided batched GEMM
      // Row-major: out[S,D] = attn[S,S] * V[S,D]
      // cuBLAS col-major mapping: C_cm = V_cm * attn_cm
      {
        float one = 1.0f;
        float zero = 0.0f;
        long long strideV = (long long)seqLen * headDim;
        long long strideLogits = (long long)seqLen * seqLen;
        if(precision == TransformerPrecision::Float32) {
          CUBLAS_ERR(name.c_str(), cublasSgemmStridedBatched(
            cudaHandles->cublas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            headDim,
            seqLen,
            seqLen,
            &one,
            reinterpret_cast<float*>(vBHSD.buf), headDim, strideV,
            reinterpret_cast<float*>(logitsBuf.buf), seqLen, strideLogits,
            &zero,
            reinterpret_cast<float*>(outBHSD.buf), headDim, strideV,
            batchSize * numHeads
          ));
        }
        else {
          cudaDataType_t dtype = transformerPrecisionToCudaType(precision);
          CUBLAS_ERR(name.c_str(), cublasGemmStridedBatchedEx(
            cudaHandles->cublas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            headDim,
            seqLen,
            seqLen,
            &one,
            vBHSD.buf, dtype, headDim, strideV,
            attnLowpBuf.buf, dtype, seqLen, strideLogits,
            &zero,
            outBHSD.buf, CUDA_R_32F, headDim, strideV,
            batchSize * numHeads,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT
          ));
        }
      }

      // Transpose attention output from [B,H,S,D] back to [B,S,H,D]
      customCudaTransposeBHSDtoBSHD(
        reinterpret_cast<float*>(outBHSD.buf), reinterpret_cast<float*>(attnBuf.buf),
        batchSize, seqLen, numHeads, headDim);
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());

      outProj.applyToLowp(cudaHandles, scratch, tokenCount, attnBuf.buf, projBuf.buf);
    }
    if(precision == TransformerPrecision::Float32)
      customCudaAddResidual(reinterpret_cast<float*>(xBuf), reinterpret_cast<float*>(projBuf.buf), tokenCount * hiddenSize);
    else if(precision == TransformerPrecision::Float16)
      customCudaAddResidual(reinterpret_cast<half*>(xBuf), reinterpret_cast<half*>(projBuf.buf), tokenCount * hiddenSize);
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
    else if(precision == TransformerPrecision::BFloat16)
      customCudaAddResidual(reinterpret_cast<bfloat16_t*>(xBuf), reinterpret_cast<bfloat16_t*>(projBuf.buf), tokenCount * hiddenSize);
#endif
    else
      ASSERT_UNREACHABLE;
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());

    norm2.apply(tokenCount, xBuf, norm2Buf.buf);
    ffnInProj.applyWithLowpInputToLowp(cudaHandles, scratch, tokenCount, norm2Buf.buf, ffnInterleavedBuf.buf);
    {
      void* ffnDstBufs[2] = {ffn1Buf.buf, ffnGateBuf.buf};
      transformerSplitCombinedLinearOutput(name, ffnInterleavedBuf.buf, tokenCount, ffnDim, 2, precision, ffnDstBufs);
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    }
    if(precision == TransformerPrecision::Float32)
      customCudaSiluMultiply(reinterpret_cast<float*>(ffn1Buf.buf), reinterpret_cast<float*>(ffnGateBuf.buf), reinterpret_cast<float*>(ffnActBuf.buf), tokenCount * ffnDim);
    else if(precision == TransformerPrecision::Float16)
      customCudaSiluMultiply(reinterpret_cast<half*>(ffn1Buf.buf), reinterpret_cast<half*>(ffnGateBuf.buf), reinterpret_cast<half*>(ffnActBuf.buf), tokenCount * ffnDim);
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
    else if(precision == TransformerPrecision::BFloat16)
      customCudaSiluMultiply(reinterpret_cast<bfloat16_t*>(ffn1Buf.buf), reinterpret_cast<bfloat16_t*>(ffnGateBuf.buf), reinterpret_cast<bfloat16_t*>(ffnActBuf.buf), tokenCount * ffnDim);
#endif
    else
      ASSERT_UNREACHABLE;
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    ffnW2.applyWithLowpInputToLowp(cudaHandles, scratch, tokenCount, ffnActBuf.buf, ffnOutBuf.buf);
    if(precision == TransformerPrecision::Float32)
      customCudaAddResidual(reinterpret_cast<float*>(xBuf), reinterpret_cast<float*>(ffnOutBuf.buf), tokenCount * hiddenSize);
    else if(precision == TransformerPrecision::Float16)
      customCudaAddResidual(reinterpret_cast<half*>(xBuf), reinterpret_cast<half*>(ffnOutBuf.buf), tokenCount * hiddenSize);
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
    else if(precision == TransformerPrecision::BFloat16)
      customCudaAddResidual(reinterpret_cast<bfloat16_t*>(xBuf), reinterpret_cast<bfloat16_t*>(ffnOutBuf.buf), tokenCount * hiddenSize);
#endif
    else
      ASSERT_UNREACHABLE;
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());
  }
};

struct TransformerModel {
  const string name;
  const int modelVersion;
  const int posLen;
  const int maxBatchSize;
  const int nnXLen;
  const int nnYLen;
  const int hiddenSize;
  const int numHeads;
  const int headDim;
  const int numInputChannels;
  const int numInputGlobalChannels;
  const int numInputMetaChannels;
  const int numPolicyChannels;
  const int numFullPolicyChannels;
  const int numValueChannels;
  const int numScoreValueChannels;
  const int numOwnershipChannels;
  const int numMiscChannels;
  const int numMoreMiscChannels;
  const int numScoringChannels;
  const int numFuturePosChannels;
  const int numSekiChannels;
  const int scoreMode;
  const int numScoreBeliefs;
  const int scoreBeliefLen;
  const int scoreBeliefProjectSize;
  const bool inputsUsingNHWC;
  const TransformerPrecision precision;

  std::unique_ptr<CudnnManager> manager;
  std::unique_ptr<TransformerStemConv> stemConv;
  std::unique_ptr<TransformerLinear> stemGlobal;
  float* posBuf;
  float* ropeCosBuf;
  float* ropeSinBuf;
  vector<std::unique_ptr<TransformerBlock>> blocks;
  std::unique_ptr<TransformerRMSNorm> finalNorm;
  std::unique_ptr<TransformerLinear> policyBoard;
  std::unique_ptr<TransformerLinear> policyPass;
  std::unique_ptr<TransformerLinear> policyBoardFull;
  std::unique_ptr<TransformerLinear> policyPassFull;
  std::unique_ptr<TransformerLinear> valueHead;
  std::unique_ptr<TransformerLinear> miscHead;
  std::unique_ptr<TransformerLinear> moreMiscHead;
  std::unique_ptr<TransformerLinear> scoreValueHead;
  std::unique_ptr<TransformerLinear> ownershipHead;
  std::unique_ptr<TransformerLinear> scoringHead;
  std::unique_ptr<TransformerLinear> futurePosHead;
  std::unique_ptr<TransformerLinear> sekiHead;
  std::unique_ptr<TransformerLinear> scoreBeliefHead;
  vector<float> scoreBeliefS2OffWeight;
  vector<float> scoreBeliefS2ParWeight;

  TransformerModel() = delete;
  TransformerModel(const TransformerModel&) = delete;
  TransformerModel& operator=(const TransformerModel&) = delete;

  TransformerModel(
    CudaHandles* cudaHandles,
    const TransformerModelDesc* desc,
    int maxBatchSz,
    int nnX,
    int nnY,
    bool inputsUseNHWC,
    TransformerPrecision precision_
  ) :
    name(desc->name),
    modelVersion(desc->modelVersion),
    posLen(desc->posLen),
    maxBatchSize(maxBatchSz),
    nnXLen(nnX),
    nnYLen(nnY),
    hiddenSize(desc->hiddenSize),
    numHeads(desc->numHeads),
    headDim(desc->headDim),
    numInputChannels(desc->numInputChannels),
    numInputGlobalChannels(desc->numInputGlobalChannels),
    numInputMetaChannels(0),
    numPolicyChannels(2),
    numFullPolicyChannels(6),
    numValueChannels(3),
    numScoreValueChannels(6),
    numOwnershipChannels(1),
    numMiscChannels(10),
    numMoreMiscChannels(8),
    numScoringChannels(1),
    numFuturePosChannels(2),
    numSekiChannels(4),
    scoreMode(desc->scoreMode),
    numScoreBeliefs(desc->numScoreBeliefs),
    scoreBeliefLen(desc->scoreBeliefLen),
    scoreBeliefProjectSize(
      desc->scoreMode == 0 ? desc->scoreBeliefLen :
      (desc->scoreBeliefLen > 0 && desc->numScoreBeliefs > 0 ? desc->scoreBeliefLen * desc->numScoreBeliefs + desc->numScoreBeliefs : 0)
    ),
    inputsUsingNHWC(inputsUseNHWC),
    precision(precision_),
    posBuf(nullptr),
    ropeCosBuf(nullptr),
    ropeSinBuf(nullptr)
  {
    if(desc->modelVersion != 15)
      throw StringError("Transformer CUDA backend 当前只支持 model version 15");
    if(nnXLen != posLen || nnYLen != posLen)
      throw StringError("Transformer CUDA backend 仅支持与导出 pos_len 完全一致的棋盘尺寸");

    int numFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
    if(numInputChannels != numFeatures)
      throw StringError("Transformer 模型的空间输入特征数量与 model version 不匹配");
    int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
    if(numInputGlobalChannels != numGlobalFeatures)
      throw StringError("Transformer 模型的全局输入特征数量与 model version 不匹配");

    manager = std::make_unique<CudnnManager>(name + "_tf", maxBatchSize, nnXLen, nnYLen);

    stemConv = std::make_unique<TransformerStemConv>(
      cudaHandles,
      name + "_stem_conv",
      maxBatchSize,
      nnXLen,
      nnYLen,
      inputsUseNHWC,
      numInputChannels,
      hiddenSize,
      desc->stemKernelSize,
      desc->stemConvWeight,
      precision
    );

    stemGlobal = std::make_unique<TransformerLinear>(cudaHandles, name + "_stem_global", numInputGlobalChannels, hiddenSize, desc->stemGlobalWeight, precision);

    if(desc->hasPosEmbed) {
      void* buf = nullptr;
      CudaUtils::mallocAndCopyToDevice(name + "_pos_embed", desc->posEmbed, buf, false);
      posBuf = reinterpret_cast<float*>(buf);
    }
    {
      void* buf = nullptr;
      CudaUtils::mallocAndCopyToDevice(name + "_rope_cos", desc->ropeCos, buf, false);
      ropeCosBuf = reinterpret_cast<float*>(buf);
    }
    {
      void* buf = nullptr;
      CudaUtils::mallocAndCopyToDevice(name + "_rope_sin", desc->ropeSin, buf, false);
      ropeSinBuf = reinterpret_cast<float*>(buf);
    }

    for(int i = 0; i < desc->numLayers; i++) {
      blocks.push_back(std::make_unique<TransformerBlock>(
        cudaHandles,
        desc->blocks[i],
        hiddenSize,
        numHeads,
        headDim,
        desc->ffnDim,
        posLen * posLen,
        maxBatchSize,
        i,
        precision
      ));
    }

    finalNorm = std::make_unique<TransformerRMSNorm>(name + "_final_norm", desc->finalNormWeight, precision);

    policyBoard = std::make_unique<TransformerLinear>(cudaHandles, name + "_policy_board", hiddenSize, 2, desc->policyBoardWeight, precision);
    policyPass = std::make_unique<TransformerLinear>(cudaHandles, name + "_policy_pass", hiddenSize, 2, desc->policyPassWeight, precision);
    if(!desc->policyBoardFullWeight.empty()) {
      policyBoardFull = std::make_unique<TransformerLinear>(cudaHandles, name + "_policy_board_full", hiddenSize, numFullPolicyChannels, desc->policyBoardFullWeight, precision);
      policyPassFull = std::make_unique<TransformerLinear>(cudaHandles, name + "_policy_pass_full", hiddenSize, numFullPolicyChannels, desc->policyPassFullWeight, precision);
      miscHead = std::make_unique<TransformerLinear>(cudaHandles, name + "_misc", hiddenSize, numMiscChannels, desc->miscWeight, precision);
      moreMiscHead = std::make_unique<TransformerLinear>(cudaHandles, name + "_moremisc", hiddenSize, numMoreMiscChannels, desc->moreMiscWeight, precision);
      scoringHead = std::make_unique<TransformerLinear>(cudaHandles, name + "_scoring", hiddenSize, numScoringChannels, desc->scoringWeight, precision);
      futurePosHead = std::make_unique<TransformerLinear>(cudaHandles, name + "_futurepos", hiddenSize, numFuturePosChannels, desc->futurePosWeight, precision);
      sekiHead = std::make_unique<TransformerLinear>(cudaHandles, name + "_seki", hiddenSize, numSekiChannels, desc->sekiWeight, precision);
      if(scoreBeliefProjectSize > 0) {
        if(scoreMode == 0)
          scoreBeliefHead = std::make_unique<TransformerLinear>(cudaHandles, name + "_scorebelief_simple", hiddenSize, scoreBeliefProjectSize, desc->scoreBeliefSimpleWeight, precision);
        else
          scoreBeliefHead = std::make_unique<TransformerLinear>(cudaHandles, name + "_scorebelief_mix", hiddenSize, scoreBeliefProjectSize, desc->scoreBeliefMixWeight, precision);
      }
      scoreBeliefS2OffWeight = desc->scoreBeliefS2OffWeight;
      scoreBeliefS2ParWeight = desc->scoreBeliefS2ParWeight;
    }
    valueHead = std::make_unique<TransformerLinear>(cudaHandles, name + "_value", hiddenSize, 3, desc->valueWeight, precision);
    scoreValueHead = std::make_unique<TransformerLinear>(cudaHandles, name + "_scorevalue", hiddenSize, 6, desc->scoreValueWeight, precision);
    ownershipHead = std::make_unique<TransformerLinear>(cudaHandles, name + "_ownership", hiddenSize, 1, desc->ownershipWeight, precision);
  }

  ~TransformerModel() {
    if(posBuf != nullptr)
      cudaFree(posBuf);
    cudaFree(ropeCosBuf);
    cudaFree(ropeSinBuf);
  }

  size_t requiredWorkspaceBytes(CudaHandles* cudaHandles, int batchSize) const {
    size_t bytes = stemConv->requiredWorkspaceBytes(cudaHandles, batchSize);
    for(const auto& block : blocks)
      bytes = std::max(bytes, block->requiredWorkspaceBytes(batchSize));
    return bytes;
  }

  bool supportsFullRawOutputs() const {
    return policyBoardFull != nullptr && policyPassFull != nullptr && miscHead != nullptr && moreMiscHead != nullptr &&
      scoringHead != nullptr && futurePosHead != nullptr && sekiHead != nullptr && scoreBeliefHead != nullptr;
  }

  const char* attentionPathName() const {
    if(blocks.empty())
      return transformerAttentionPathName(TransformerAttentionPath::Legacy);
    bool sawOfficial = false;
    bool sawLegacy = false;
    for(const auto& block : blocks) {
      if(block->usesOfficialAttention())
        sawOfficial = true;
      else
        sawLegacy = true;
    }
    if(sawOfficial && sawLegacy)
      return "mixed";
    if(sawOfficial)
      return transformerAttentionPathName(TransformerAttentionPath::CudnnSdpa);
    return transformerAttentionPathName(TransformerAttentionPath::Legacy);
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    float* inputBuf,
    float* inputGlobalBuf,
    float* policyPassBuf,
    float* policyBuf,
    float* valueBuf,
    float* scoreValueBuf,
    float* ownershipBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    int seqLen = posLen * posLen;
    int tokenCount = batchSize * seqLen;
    size_t stemBytes = (size_t)batchSize * hiddenSize * seqLen * sizeof(float);
    size_t globalBytes = (size_t)batchSize * hiddenSize * sizeof(float);
    size_t residentBytes = (size_t)tokenCount * hiddenSize * transformerPrecisionElemSize(precision);
    size_t pooledBytes = (size_t)batchSize * hiddenSize * transformerPrecisionElemSize(precision);

    SizedBuf<void*> stemSpatialBuf(scratch->allocator, stemBytes);
    SizedBuf<void*> stemGlobalBuf(scratch->allocator, globalBytes);
    SizedBuf<void*> xBuf(scratch->allocator, residentBytes);
    SizedBuf<void*> pooledBuf(scratch->allocator, pooledBytes);

    stemConv->apply(cudaHandles, scratch, batchSize, inputBuf, reinterpret_cast<float*>(stemSpatialBuf.buf), workspaceBuf, workspaceBytes);
    stemGlobal->apply(cudaHandles, scratch, batchSize, inputGlobalBuf, stemGlobalBuf.buf);
    if(precision == TransformerPrecision::Float32)
      customCudaNCHWToNLCAddBiasPos(reinterpret_cast<float*>(stemSpatialBuf.buf), reinterpret_cast<float*>(stemGlobalBuf.buf), posBuf, reinterpret_cast<float*>(xBuf.buf), batchSize, hiddenSize, posLen, posLen);
    else if(precision == TransformerPrecision::Float16)
      customCudaNCHWToNLCAddBiasPos(reinterpret_cast<float*>(stemSpatialBuf.buf), reinterpret_cast<float*>(stemGlobalBuf.buf), posBuf, reinterpret_cast<half*>(xBuf.buf), batchSize, hiddenSize, posLen, posLen);
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
    else if(precision == TransformerPrecision::BFloat16)
      customCudaNCHWToNLCAddBiasPos(reinterpret_cast<float*>(stemSpatialBuf.buf), reinterpret_cast<float*>(stemGlobalBuf.buf), posBuf, reinterpret_cast<bfloat16_t*>(xBuf.buf), batchSize, hiddenSize, posLen, posLen);
#endif
    else
      ASSERT_UNREACHABLE;
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());

    for(const auto& block : blocks)
      block->apply(cudaHandles, scratch, batchSize, ropeCosBuf, ropeSinBuf, xBuf.buf, workspaceBuf, workspaceBytes);

    finalNorm->apply(tokenCount, xBuf.buf, xBuf.buf);

    policyBoard->applyWithLowpInput(cudaHandles, scratch, tokenCount, xBuf.buf, policyBuf);
    if(precision == TransformerPrecision::Float32)
      customCudaMeanPoolNLC(reinterpret_cast<float*>(xBuf.buf), reinterpret_cast<float*>(pooledBuf.buf), batchSize, seqLen, hiddenSize);
    else if(precision == TransformerPrecision::Float16)
      customCudaMeanPoolNLC(reinterpret_cast<half*>(xBuf.buf), reinterpret_cast<half*>(pooledBuf.buf), batchSize, seqLen, hiddenSize);
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
    else if(precision == TransformerPrecision::BFloat16)
      customCudaMeanPoolNLC(reinterpret_cast<bfloat16_t*>(xBuf.buf), reinterpret_cast<bfloat16_t*>(pooledBuf.buf), batchSize, seqLen, hiddenSize);
#endif
    else
      ASSERT_UNREACHABLE;
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    policyPass->applyWithLowpInput(cudaHandles, scratch, batchSize, pooledBuf.buf, policyPassBuf);
    valueHead->applyWithLowpInput(cudaHandles, scratch, batchSize, pooledBuf.buf, valueBuf);
    scoreValueHead->applyWithLowpInput(cudaHandles, scratch, batchSize, pooledBuf.buf, scoreValueBuf);
    ownershipHead->applyWithLowpInput(cudaHandles, scratch, tokenCount, xBuf.buf, ownershipBuf);
  }

  void applyFull(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    float* inputBuf,
    float* inputGlobalBuf,
    float* policyPassFullBuf,
    float* policyFullBuf,
    float* valueBuf,
    float* miscBuf,
    float* moreMiscBuf,
    float* ownershipBuf,
    float* scoringBuf,
    float* futurePosBuf,
    float* sekiBuf,
    float* scoreBeliefProjectBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    if(!supportsFullRawOutputs())
      throw StringError("Transformer CUDA backend full raw outputs unavailable: model file lacks full head weights");

    int seqLen = posLen * posLen;
    int tokenCount = batchSize * seqLen;
    size_t stemBytes = (size_t)batchSize * hiddenSize * seqLen * sizeof(float);
    size_t globalBytes = (size_t)batchSize * hiddenSize * sizeof(float);
    size_t residentBytes = (size_t)tokenCount * hiddenSize * transformerPrecisionElemSize(precision);
    size_t pooledBytes = (size_t)batchSize * hiddenSize * transformerPrecisionElemSize(precision);

    SizedBuf<void*> stemSpatialBuf(scratch->allocator, stemBytes);
    SizedBuf<void*> stemGlobalBuf(scratch->allocator, globalBytes);
    SizedBuf<void*> xBuf(scratch->allocator, residentBytes);
    SizedBuf<void*> pooledBuf(scratch->allocator, pooledBytes);

    stemConv->apply(cudaHandles, scratch, batchSize, inputBuf, reinterpret_cast<float*>(stemSpatialBuf.buf), workspaceBuf, workspaceBytes);
    stemGlobal->apply(cudaHandles, scratch, batchSize, inputGlobalBuf, stemGlobalBuf.buf);
    if(precision == TransformerPrecision::Float32)
      customCudaNCHWToNLCAddBiasPos(reinterpret_cast<float*>(stemSpatialBuf.buf), reinterpret_cast<float*>(stemGlobalBuf.buf), posBuf, reinterpret_cast<float*>(xBuf.buf), batchSize, hiddenSize, posLen, posLen);
    else if(precision == TransformerPrecision::Float16)
      customCudaNCHWToNLCAddBiasPos(reinterpret_cast<float*>(stemSpatialBuf.buf), reinterpret_cast<float*>(stemGlobalBuf.buf), posBuf, reinterpret_cast<half*>(xBuf.buf), batchSize, hiddenSize, posLen, posLen);
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
    else if(precision == TransformerPrecision::BFloat16)
      customCudaNCHWToNLCAddBiasPos(reinterpret_cast<float*>(stemSpatialBuf.buf), reinterpret_cast<float*>(stemGlobalBuf.buf), posBuf, reinterpret_cast<bfloat16_t*>(xBuf.buf), batchSize, hiddenSize, posLen, posLen);
#endif
    else
      ASSERT_UNREACHABLE;
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());

    for(const auto& block : blocks)
      block->apply(cudaHandles, scratch, batchSize, ropeCosBuf, ropeSinBuf, xBuf.buf, workspaceBuf, workspaceBytes);

    finalNorm->apply(tokenCount, xBuf.buf, xBuf.buf);

    policyBoardFull->applyWithLowpInput(cudaHandles, scratch, tokenCount, xBuf.buf, policyFullBuf);
    ownershipHead->applyWithLowpInput(cudaHandles, scratch, tokenCount, xBuf.buf, ownershipBuf);
    scoringHead->applyWithLowpInput(cudaHandles, scratch, tokenCount, xBuf.buf, scoringBuf);
    futurePosHead->applyWithLowpInput(cudaHandles, scratch, tokenCount, xBuf.buf, futurePosBuf);
    sekiHead->applyWithLowpInput(cudaHandles, scratch, tokenCount, xBuf.buf, sekiBuf);
    if(precision == TransformerPrecision::Float32)
      customCudaMeanPoolNLC(reinterpret_cast<float*>(xBuf.buf), reinterpret_cast<float*>(pooledBuf.buf), batchSize, seqLen, hiddenSize);
    else if(precision == TransformerPrecision::Float16)
      customCudaMeanPoolNLC(reinterpret_cast<half*>(xBuf.buf), reinterpret_cast<half*>(pooledBuf.buf), batchSize, seqLen, hiddenSize);
#ifdef KATAGO_CUDA_BFLOAT16_AVAILABLE
    else if(precision == TransformerPrecision::BFloat16)
      customCudaMeanPoolNLC(reinterpret_cast<bfloat16_t*>(xBuf.buf), reinterpret_cast<bfloat16_t*>(pooledBuf.buf), batchSize, seqLen, hiddenSize);
#endif
    else
      ASSERT_UNREACHABLE;
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    policyPassFull->applyWithLowpInput(cudaHandles, scratch, batchSize, pooledBuf.buf, policyPassFullBuf);
    valueHead->applyWithLowpInput(cudaHandles, scratch, batchSize, pooledBuf.buf, valueBuf);
    miscHead->applyWithLowpInput(cudaHandles, scratch, batchSize, pooledBuf.buf, miscBuf);
    moreMiscHead->applyWithLowpInput(cudaHandles, scratch, batchSize, pooledBuf.buf, moreMiscBuf);
    scoreBeliefHead->applyWithLowpInput(cudaHandles, scratch, batchSize, pooledBuf.buf, scoreBeliefProjectBuf);
  }
};

static void logSoftmax1DInplace(vector<float>& values) {
  if(values.empty())
    return;
  float maxVal = values[0];
  for(size_t i = 1; i < values.size(); i++)
    maxVal = std::max(maxVal, values[i]);
  float sum = 0.0f;
  for(size_t i = 0; i < values.size(); i++)
    sum += std::exp(values[i] - maxVal);
  float logSum = maxVal + std::log(sum);
  for(size_t i = 0; i < values.size(); i++)
    values[i] -= logSum;
}

static void finalizeTransformerScoreBelief(
  const TransformerModel& model,
  const float* globalInput,
  const vector<float>& scoreBeliefProject,
  float* scoreBeliefOut
) {
  if(model.scoreMode == 0) {
    vector<float> logits = scoreBeliefProject;
    logSoftmax1DInplace(logits);
    std::copy(logits.begin(), logits.end(), scoreBeliefOut);
    return;
  }

  const int len = model.scoreBeliefLen;
  const int numBeliefs = model.numScoreBeliefs;
  vector<float> belief(scoreBeliefProject.begin(), scoreBeliefProject.begin() + (size_t)len * numBeliefs);
  vector<float> mixLogits(scoreBeliefProject.begin() + (size_t)len * numBeliefs, scoreBeliefProject.end());

  if(model.scoreMode == 2) {
    const int mid = len / 2;
    const float scoreParity = globalInput[model.numInputGlobalChannels - 1];
    for(int i = 0; i < len; i++) {
      int diff = i - mid;
      int parityBit = ((diff % 2) + 2) % 2;
      float offsetTerm = 0.05f * ((float)diff + 0.5f);
      float parityTerm = (0.5f - (float)parityBit) * scoreParity;
      for(int j = 0; j < numBeliefs; j++) {
        belief[(size_t)i * numBeliefs + j] +=
          offsetTerm * model.scoreBeliefS2OffWeight[j] +
          parityTerm * model.scoreBeliefS2ParWeight[j];
      }
    }
  }

  for(int j = 0; j < numBeliefs; j++) {
    float maxVal = belief[j];
    for(int i = 1; i < len; i++)
      maxVal = std::max(maxVal, belief[(size_t)i * numBeliefs + j]);
    float sum = 0.0f;
    for(int i = 0; i < len; i++)
      sum += std::exp(belief[(size_t)i * numBeliefs + j] - maxVal);
    float logSum = maxVal + std::log(sum);
    for(int i = 0; i < len; i++)
      belief[(size_t)i * numBeliefs + j] -= logSum;
  }

  logSoftmax1DInplace(mixLogits);
  for(int i = 0; i < len; i++) {
    float maxVal = belief[(size_t)i * numBeliefs] + mixLogits[0];
    for(int j = 1; j < numBeliefs; j++)
      maxVal = std::max(maxVal, belief[(size_t)i * numBeliefs + j] + mixLogits[j]);
    float sum = 0.0f;
    for(int j = 0; j < numBeliefs; j++)
      sum += std::exp(belief[(size_t)i * numBeliefs + j] + mixLogits[j] - maxVal);
    scoreBeliefOut[i] = maxVal + std::log(sum);
  }
}


//------------------------------------------------------------------------------

struct LoadedModel {
  bool isTransformer;
  ModelDesc modelDesc;
  std::unique_ptr<TransformerModelDesc> transformerDesc;

  LoadedModel(const string& fileName, const string& expectedSha256) {
    transformerDesc = std::make_unique<TransformerModelDesc>();
    if(TransformerModelDesc::tryLoadFromFileMaybeGZipped(fileName, expectedSha256, *transformerDesc)) {
      isTransformer = true;
      modelDesc.name = transformerDesc->name;
      modelDesc.sha256 = transformerDesc->sha256;
      modelDesc.modelVersion = transformerDesc->modelVersion;
      modelDesc.numInputChannels = transformerDesc->numInputChannels;
      modelDesc.numInputGlobalChannels = transformerDesc->numInputGlobalChannels;
      modelDesc.numInputMetaChannels = 0;
      modelDesc.numPolicyChannels = 2;
      modelDesc.numValueChannels = 3;
      modelDesc.numScoreValueChannels = 6;
      modelDesc.numOwnershipChannels = 1;
      modelDesc.metaEncoderVersion = 0;
      modelDesc.postProcessParams = transformerDesc->postProcessParams;
      modelDesc.trunk.trunkNumChannels = transformerDesc->hiddenSize;
      modelDesc.trunk.numBlocks = transformerDesc->numLayers;
      modelDesc.trunk.initialConv.convXSize = transformerDesc->stemKernelSize;
      modelDesc.trunk.initialConv.convYSize = transformerDesc->stemKernelSize;
      modelDesc.trunk.initialConv.inChannels = transformerDesc->numInputChannels;
      modelDesc.trunk.initialConv.outChannels = transformerDesc->hiddenSize;
    }
    else {
      isTransformer = false;
      transformerDesc.reset();
      ModelDesc::loadFromFileMaybeGZipped(fileName,modelDesc,expectedSha256);
      modelDesc.applyScale8ToReduceActivations();
    }
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  LoadedModel* loadedModel = new LoadedModel(file,expectedSha256);
  return loadedModel;
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

bool NeuralNet::isTransformerModel(const LoadedModel* loadedModel) {
  return loadedModel->isTransformer;
}

//------------------------------------------------------------------------------

struct Buffers {
  //All of these are device pointers

  float* inputBufFloat;
  void* inputBuf;
  float* inputGlobalBufFloat;
  void* inputGlobalBuf;
  float* inputMetaBufFloat;
  void* inputMetaBuf;
  size_t inputBufBytesFloat;
  size_t inputBufBytes;
  size_t inputGlobalBufBytesFloat;
  size_t inputGlobalBufBytes;
  size_t inputMetaBufBytesFloat;
  size_t inputMetaBufBytes;

  float* policyPassBuf;
  size_t policyPassBufBytes;
  float* policyBuf;
  size_t policyBufBytes;

  float* valueBuf;
  size_t valueBufBytes;
  float* scoreValueBuf;
  size_t scoreValueBufBytes;
  void* ownershipBuf;
  size_t ownershipBufBytes;

  void* workspaceBuf;
  size_t workspaceBytes;

  Buffers() = delete;
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;

  Buffers(CudaHandles* cudaHandles, const Model& m, const ScratchBuffers& scratch) {
    size_t batchXYFloatBytes = (size_t)scratch.batchXYFloatBytes;
    size_t batchFloatBytes = (size_t)scratch.batchFloatBytes;
    size_t batchXYBytes = (size_t)scratch.batchXYBytes;
    size_t batchBytes = (size_t)scratch.batchBytes;

    inputBufBytesFloat = m.numInputChannels * batchXYFloatBytes;
    inputBufBytes = m.numInputChannels * batchXYBytes;
    inputGlobalBufBytesFloat = m.numInputGlobalChannels * batchFloatBytes;
    inputGlobalBufBytes = m.numInputGlobalChannels * batchBytes;
    inputMetaBufBytesFloat = m.numInputMetaChannels * batchFloatBytes;
    inputMetaBufBytes = m.numInputMetaChannels * batchBytes;

    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&inputBufFloat), inputBufBytesFloat));
    CUDA_ERR("Buffers",cudaMalloc(&inputBuf, inputBufBytes));
    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&inputGlobalBufFloat), inputGlobalBufBytesFloat));
    CUDA_ERR("Buffers",cudaMalloc(&inputGlobalBuf, inputGlobalBufBytes));
    if(m.numInputMetaChannels > 0) {
      CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&inputMetaBufFloat), inputMetaBufBytesFloat));
      CUDA_ERR("Buffers",cudaMalloc(&inputMetaBuf, inputMetaBufBytes));
    }
    else {
      inputMetaBufFloat = NULL;
      inputMetaBuf = NULL;
    }

    if(m.modelVersion >= 16)
      testAssert(m.policyHead->p2Channels == 4);
    else if(m.modelVersion >= 12)
      testAssert(m.policyHead->p2Channels == 2);
    else
      testAssert(m.policyHead->p2Channels == 1);

    policyPassBufBytes = m.policyHead->p2Channels * batchFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&policyPassBuf), policyPassBufBytes));
    policyBufBytes = m.policyHead->p2Channels * batchXYFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&policyBuf), policyBufBytes));

    valueBufBytes = m.valueHead->valueChannels * batchFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&valueBuf), valueBufBytes));

    scoreValueBufBytes = m.valueHead->scoreValueChannels * batchFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&scoreValueBuf), scoreValueBufBytes));

    //This buf is used for both an intermdiate fp16 result in fp16 mode, and ALSO the final fp32 output, so always must be fp32-sized
    ownershipBufBytes = m.valueHead->ownershipChannels * batchXYFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(&ownershipBuf, ownershipBufBytes));

    //In theory the requiredWorkspaceBytes calls could give us values non-monotone in batch size
    //such as if the convolution algorithm changes between batch size 1 and larger.
    //So we call it for all the batch sizes.
    size_t bytes = 0;
    size_t b;
    for(int batchSize = 1; batchSize <= m.maxBatchSize; batchSize++) {
      b = m.requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }

    CUDA_ERR("Buffers",cudaMalloc(&workspaceBuf, bytes));
    workspaceBytes = bytes;
  }

  ~Buffers() {
    cudaFree(inputBufFloat);
    cudaFree(inputBuf);
    cudaFree(inputGlobalBufFloat);
    cudaFree(inputGlobalBuf);
    if(inputMetaBufFloat != NULL)
      cudaFree(inputMetaBufFloat);
    if(inputMetaBuf != NULL)
      cudaFree(inputMetaBuf);

    cudaFree(policyPassBuf);
    cudaFree(policyBuf);

    cudaFree(valueBuf);
    cudaFree(scoreValueBuf);
    cudaFree(ownershipBuf);

    cudaFree(workspaceBuf);
  }

};

//------------------------------------------------------------------------------

struct TransformerBuffers {
  float* inputBuf;
  float* inputGlobalBuf;
  float* policyPassBuf;
  float* policyBuf;
  float* valueBuf;
  float* scoreValueBuf;
  float* ownershipBuf;
  void* workspaceBuf;

  size_t inputBufBytes;
  size_t inputGlobalBufBytes;
  size_t policyPassBufBytes;
  size_t policyBufBytes;
  size_t valueBufBytes;
  size_t scoreValueBufBytes;
  size_t ownershipBufBytes;
  size_t workspaceBytes;

  TransformerBuffers() = delete;
  TransformerBuffers(const TransformerBuffers&) = delete;
  TransformerBuffers& operator=(const TransformerBuffers&) = delete;

  TransformerBuffers(CudaHandles* cudaHandles, const TransformerModel& m, int maxBatchSize) {
    size_t seqLen = (size_t)m.nnXLen * m.nnYLen;

    inputBufBytes = (size_t)m.numInputChannels * maxBatchSize * seqLen * sizeof(float);
    inputGlobalBufBytes = (size_t)m.numInputGlobalChannels * maxBatchSize * sizeof(float);
    policyPassBufBytes = (size_t)m.numPolicyChannels * maxBatchSize * sizeof(float);
    policyBufBytes = (size_t)m.numPolicyChannels * maxBatchSize * seqLen * sizeof(float);
    valueBufBytes = (size_t)m.numValueChannels * maxBatchSize * sizeof(float);
    scoreValueBufBytes = (size_t)m.numScoreValueChannels * maxBatchSize * sizeof(float);
    ownershipBufBytes = (size_t)m.numOwnershipChannels * maxBatchSize * seqLen * sizeof(float);

    CUDA_ERR("TransformerBuffers", cudaMalloc(reinterpret_cast<void**>(&inputBuf), inputBufBytes));
    CUDA_ERR("TransformerBuffers", cudaMalloc(reinterpret_cast<void**>(&inputGlobalBuf), inputGlobalBufBytes));
    CUDA_ERR("TransformerBuffers", cudaMalloc(reinterpret_cast<void**>(&policyPassBuf), policyPassBufBytes));
    CUDA_ERR("TransformerBuffers", cudaMalloc(reinterpret_cast<void**>(&policyBuf), policyBufBytes));
    CUDA_ERR("TransformerBuffers", cudaMalloc(reinterpret_cast<void**>(&valueBuf), valueBufBytes));
    CUDA_ERR("TransformerBuffers", cudaMalloc(reinterpret_cast<void**>(&scoreValueBuf), scoreValueBufBytes));
    CUDA_ERR("TransformerBuffers", cudaMalloc(reinterpret_cast<void**>(&ownershipBuf), ownershipBufBytes));

    size_t bytes = 0;
    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      bytes = std::max(bytes, m.requiredWorkspaceBytes(cudaHandles, batchSize));
    }
    workspaceBytes = bytes;
    if(workspaceBytes > 0) {
      CUDA_ERR("TransformerBuffers", cudaMalloc(&workspaceBuf, workspaceBytes));
    }
    else
      workspaceBuf = nullptr;
  }

  ~TransformerBuffers() {
    cudaFree(inputBuf);
    cudaFree(inputGlobalBuf);
    cudaFree(policyPassBuf);
    cudaFree(policyBuf);
    cudaFree(valueBuf);
    cudaFree(scoreValueBuf);
    cudaFree(ownershipBuf);
    if(workspaceBuf != nullptr)
      cudaFree(workspaceBuf);
  }
};

//------------------------------------------------------------------------------

struct ComputeContext {
  int nnXLen;
  int nnYLen;
  enabled_t useFP16Mode;
  compute_precision_t precisionMode;
  enabled_t useNHWCMode;
};

ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& openCLTunerFile,
  const string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  compute_precision_t precisionMode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel
) {
  (void)gpuIdxs;
  (void)logger;
  (void)openCLTunerFile;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;
  (void)loadedModel;

  ComputeContext* context = new ComputeContext();
  context->nnXLen = nnXLen;
  context->nnYLen = nnYLen;
  context->useFP16Mode = useFP16Mode;
  context->precisionMode = precisionMode;
  context->useNHWCMode = useNHWCMode;
  return context;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//------------------------------------------------------------------------------

struct ComputeHandle {
  std::unique_ptr<CudaHandles> cudaHandles;
  bool isTransformer;
  std::unique_ptr<Model> model;
  std::unique_ptr<TransformerModel> transformerModel;
  std::unique_ptr<ScratchBuffers> scratch;
  std::unique_ptr<Buffers> buffers;
  std::unique_ptr<TransformerBuffers> transformerBuffers;
  const bool usingFP16;
  const TransformerPrecision transformerPrecision;
  const int nnXLen;
  const int nnYLen;
  const bool requireExactNNLen;
  const bool inputsUseNHWC;
  const bool usingNHWC;

  ComputeHandle(
    const ComputeContext* context,
    const LoadedModel* loadedModel,
    int majorComputeCapability,
    int minorComputeCapability,
    int maxBatchSize,
    bool requireExactNNLen_,
    bool inputsUseNHWC_,
    bool useFP16,
    bool useNHWC,
    TransformerPrecision transformerPrecision_
  ) :
    isTransformer(loadedModel->isTransformer),
    usingFP16(isTransformer ? transformerPrecision_ == TransformerPrecision::Float16 : useFP16),
    transformerPrecision(transformerPrecision_),
    nnXLen(context->nnXLen),
    nnYLen(context->nnYLen),
    requireExactNNLen(requireExactNNLen_),
    inputsUseNHWC(inputsUseNHWC_),
    usingNHWC(useNHWC)
  {
    cudaHandles = std::make_unique<CudaHandles>(majorComputeCapability,minorComputeCapability);
    if(!isTransformer) {
      model = std::make_unique<Model>(
        cudaHandles.get(), &(loadedModel->modelDesc), maxBatchSize,
        nnXLen, nnYLen, inputsUseNHWC, useFP16, useNHWC
      );
      scratch = std::make_unique<ScratchBuffers>(maxBatchSize, nnXLen, nnYLen, useFP16);
      buffers = std::make_unique<Buffers>(cudaHandles.get(), *model, *scratch);
    }
    else {
      transformerModel = std::make_unique<TransformerModel>(
        cudaHandles.get(), loadedModel->transformerDesc.get(), maxBatchSize,
        nnXLen, nnYLen, inputsUseNHWC, transformerPrecision
      );
      scratch = std::make_unique<ScratchBuffers>(maxBatchSize, nnXLen, nnYLen, false);
      transformerBuffers = std::make_unique<TransformerBuffers>(cudaHandles.get(), *transformerModel, maxBatchSize);
    }

    //Synchronize after creating buffers and copying all the weights, just in case
    CUDA_ERR("ComputeHandle", cudaDeviceSynchronize());
  }
  ~ComputeHandle() {
  }

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;
};

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  //Use whatever CUDA believes GPU 0 to be.
  if(gpuIdxForThisThread == -1)
    gpuIdxForThisThread = 0;

  CUDA_ERR("createComputeHandle",cudaSetDevice(gpuIdxForThisThread));

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop,gpuIdxForThisThread);

  bool useFP16 = false;
  bool useNHWC = false;
  TransformerPrecision transformerPrecision = TransformerPrecision::Float32;
  if(loadedModel->isTransformer) {
    transformerPrecision = chooseTransformerPrecision(context->precisionMode, prop.major, prop.minor);
    useFP16 = (transformerPrecision == TransformerPrecision::Float16);
    useNHWC = false;
    if(!requireExactNNLen) {
      throw StringError("Transformer CUDA backend 仅支持 requireExactNNLen=true，且棋盘尺寸必须固定等于模型 pos_len");
    }
  }
  else {
    if(context->precisionMode == compute_precision_t::BF16)
      throw StringError("Cuda CNN backend 暂不支持 bf16 precision");

    enabled_t effectiveUseFP16Mode = context->useFP16Mode;
    if(context->precisionMode == compute_precision_t::FP16)
      effectiveUseFP16Mode = enabled_t::True;
    else if(context->precisionMode == compute_precision_t::FP32)
      effectiveUseFP16Mode = enabled_t::False;

    //Old GPUs - use FP32 and explicitly fail if FP16 enabled
    if(prop.major < 5 || (prop.major == 5 && prop.minor < 3)) {
      if(effectiveUseFP16Mode == enabled_t::True)
        throw StringError("Cuda device versions below 5.3 do not support useFP16=true");
      if(context->useNHWCMode == enabled_t::True)
        useNHWC = true;
    }
    //In theory these GPUs support FP16, so allow if the user wants.
    else if(prop.major < 6) {
      if(effectiveUseFP16Mode == enabled_t::True)
        useFP16 = true;
      if(context->useNHWCMode == enabled_t::True)
        useNHWC = true;
    }
    //On Pascal architecture, default to using FP16 operations
    //Actually, just use FP32 - there's a risk that on certain cards this might just be a lot worse.
    //A user manually fine-tuning for performance can just enable it themselves if they know how.
    else if(prop.major < 7) {
      if(effectiveUseFP16Mode == enabled_t::True)
        useFP16 = true;
      if(context->useNHWCMode == enabled_t::True)
        useNHWC = true;
    }
    //On Volta and higher, use FP16 and NHWC together because we have tensor cores.
    else {
      if(effectiveUseFP16Mode == enabled_t::True || effectiveUseFP16Mode == enabled_t::Auto)
        useFP16 = true;
      if(context->useNHWCMode == enabled_t::True || (context->useNHWCMode == enabled_t::Auto && useFP16))
        useNHWC = true;
    }
  }

  if(
    loadedModel->isTransformer &&
    transformerPrecision == TransformerPrecision::Float16
  ) {
    const string warning =
      "WARNING: CUDA Transformer fp16 inference may overflow to NaN/Inf on some models or positions. "
      "This is more likely when the checkpoint was trained with bf16/fp32 or mixed precision. "
      "Validate raw outputs before relying on fp16.";
    if(logger != NULL)
      logger->write(warning);
    else
      cerr << warning << endl;
  }

  ComputeHandle* gpuHandle = new ComputeHandle(
    context,loadedModel,prop.major,prop.minor,maxBatchSize,requireExactNNLen,inputsUseNHWC,useFP16,useNHWC,transformerPrecision
  );
  if(logger != NULL) {
    logger->write(
      "Cuda backend thread " + Global::intToString(serverThreadIdx) + ": Found GPU " + string(prop.name)
      + " memory " + Global::uint64ToString(prop.totalGlobalMem)
      + " compute capability major " + Global::intToString(prop.major)
      + " minor " + Global::intToString(prop.minor)
    );
    logger->write(
      "Cuda backend thread " + Global::intToString(serverThreadIdx) + ": Model version " + Global::intToString(loadedModel->modelDesc.modelVersion) +
      " useFP16 = " + Global::boolToString(useFP16) +
      " useNHWC = " + Global::boolToString(useNHWC) +
      " modelType = " + string(loadedModel->isTransformer ? "transformer" : "cnn") +
      (loadedModel->isTransformer ?
        " transformerPrecision = " + string(transformerPrecisionName(transformerPrecision)) +
        " transformerAttention = " + string(gpuHandle->transformerModel->attentionPathName())
        : "")
    );
    logger->write(
      "Cuda backend thread " + Global::intToString(serverThreadIdx) + ": Model name: " + loadedModel->modelDesc.name
    );
  }
  return gpuHandle;
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  return handle->usingFP16;
}

//------------------------------------------------------------------------------

void NeuralNet::printDevices() {
  int numDevices = 0;
  cudaGetDeviceCount(&numDevices);
  for(int i = 0; i<numDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    cout << "Found CUDA device " << i << ": " << prop.name << endl;
  }
}


//------------------------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleInputElts;
  size_t singleInputBytes;
  size_t singleInputGlobalElts;
  size_t singleInputGlobalBytes;
  size_t singleInputMetaElts;
  size_t singleInputMetaBytes;
  size_t singlePolicyPassResultElts;
  size_t singlePolicyPassResultBytes;
  size_t singlePolicyResultElts;
  size_t singlePolicyResultBytes;
  size_t singleValueResultElts;
  size_t singleValueResultBytes;
  size_t singleScoreValueResultElts;
  size_t singleScoreValueResultBytes;
  size_t singleOwnershipResultElts;
  size_t singleOwnershipResultBytes;

  size_t userInputBufferBytes;
  size_t userInputGlobalBufferBytes;
  size_t userInputMetaBufferBytes;
  size_t policyPassResultBufferBytes;
  size_t policyResultBufferBytes;
  size_t valueResultBufferBytes;
  size_t scoreValueResultBufferBytes;
  size_t ownershipResultBufferBytes;

  float* userInputBuffer; //Host pointer
  float* userInputGlobalBuffer; //Host pointer
  float* userInputMetaBuffer; //Host pointer

  float* policyPassResults; //Host pointer
  float* policyResults; //Host pointer
  float* valueResults; //Host pointer
  float* scoreValueResults; //Host pointer
  float* ownershipResults; //Host pointer

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    maxBatchSize = maxBatchSz;
    singleInputElts = (size_t)m.numInputChannels * nnXLen * nnYLen;
    singleInputBytes = (size_t)m.numInputChannels * nnXLen * nnYLen * sizeof(float);
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    singleInputGlobalBytes = (size_t)m.numInputGlobalChannels * sizeof(float);
    singleInputMetaElts = (size_t)m.numInputMetaChannels;
    singleInputMetaBytes = (size_t)m.numInputMetaChannels * sizeof(float);
    singlePolicyPassResultElts = (size_t)(m.numPolicyChannels);
    singlePolicyPassResultBytes = (size_t)(m.numPolicyChannels) * sizeof(float);
    singlePolicyResultElts = (size_t)(m.numPolicyChannels * nnXLen * nnYLen);
    singlePolicyResultBytes = (size_t)(m.numPolicyChannels * nnXLen * nnYLen) * sizeof(float);
    singleValueResultElts = (size_t)m.numValueChannels;
    singleValueResultBytes = (size_t)m.numValueChannels * sizeof(float);
    singleScoreValueResultElts = (size_t)m.numScoreValueChannels;
    singleScoreValueResultBytes = (size_t)m.numScoreValueChannels * sizeof(float);
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * nnXLen * nnYLen;
    singleOwnershipResultBytes = (size_t)m.numOwnershipChannels * nnXLen * nnYLen * sizeof(float);

    assert(NNModelVersion::getNumSpatialFeatures(m.modelVersion) == m.numInputChannels);
    assert(NNModelVersion::getNumGlobalFeatures(m.modelVersion) == m.numInputGlobalChannels);
    if(m.numInputMetaChannels > 0) {
      assert(SGFMetadata::METADATA_INPUT_NUM_CHANNELS == m.numInputMetaChannels);
    }

    userInputBufferBytes = (size_t)m.numInputChannels * maxBatchSize * nnXLen * nnYLen * sizeof(float);
    userInputGlobalBufferBytes = (size_t)m.numInputGlobalChannels * maxBatchSize * sizeof(float);
    userInputMetaBufferBytes = (size_t)m.numInputMetaChannels * maxBatchSize * sizeof(float);
    policyPassResultBufferBytes = (size_t)maxBatchSize * m.numPolicyChannels * sizeof(float);
    policyResultBufferBytes = (size_t)maxBatchSize * m.numPolicyChannels * nnXLen * nnYLen * sizeof(float);
    valueResultBufferBytes = (size_t)maxBatchSize * m.numValueChannels * sizeof(float);
    scoreValueResultBufferBytes = (size_t)maxBatchSize * m.numScoreValueChannels * sizeof(float);
    ownershipResultBufferBytes = (size_t)maxBatchSize * nnXLen * nnYLen * m.numOwnershipChannels * sizeof(float);

    userInputBuffer = new float[(size_t)m.numInputChannels * maxBatchSize * nnXLen * nnYLen];
    userInputGlobalBuffer = new float[(size_t)m.numInputGlobalChannels * maxBatchSize];
    if(m.numInputMetaChannels > 0)
      userInputMetaBuffer = new float[(size_t)m.numInputMetaChannels * maxBatchSize];
    else
      userInputMetaBuffer = NULL;

    policyPassResults = new float[(size_t)maxBatchSize * m.numPolicyChannels];
    policyResults = new float[(size_t)maxBatchSize * m.numPolicyChannels * nnXLen * nnYLen];
    valueResults = new float[(size_t)maxBatchSize * m.numValueChannels];

    scoreValueResults = new float[(size_t)maxBatchSize * m.numScoreValueChannels];
    ownershipResults = new float[(size_t)maxBatchSize * nnXLen * nnYLen * m.numOwnershipChannels];
  }

  ~InputBuffers() {
    delete[] userInputBuffer;
    delete[] userInputGlobalBuffer;
    if(userInputMetaBuffer != NULL)
      delete[] userInputMetaBuffer;
    delete[] policyPassResults;
    delete[] policyResults;
    delete[] valueResults;
    delete[] scoreValueResults;
    delete[] ownershipResults;
  }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;

};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel,maxBatchSize,nnXLen,nnYLen);
}
void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

//---------------------------------------------------------------------------------------


void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);
  const int batchSize = numBatchEltsFilled;
  const int nnXLen = gpuHandle->nnXLen;
  const int nnYLen = gpuHandle->nnYLen;
  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];
  if(gpuHandle->isTransformer) {
    const int modelVersion = gpuHandle->transformerModel->modelVersion;
    const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
    const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
    const int numPolicyChannels = gpuHandle->transformerModel->numPolicyChannels;
    assert(numSpatialFeatures == gpuHandle->transformerModel->numInputChannels);
    assert(numGlobalFeatures == gpuHandle->transformerModel->numInputGlobalChannels);
    assert(inputBuffers->singleInputElts == (size_t)numSpatialFeatures * nnXLen * nnYLen);
    assert(inputBuffers->singleInputBytes == inputBuffers->singleInputElts * sizeof(float));
    assert(inputBuffers->singleInputGlobalElts == (size_t)numGlobalFeatures);
    assert(inputBuffers->singleInputGlobalBytes == inputBuffers->singleInputGlobalElts * sizeof(float));
    assert(inputBuffers->singleInputMetaElts == 0);
    assert(inputBuffers->singleInputMetaBytes == 0);
    assert(inputBuffers->singlePolicyPassResultElts == (size_t)numPolicyChannels);
    assert(inputBuffers->singlePolicyPassResultBytes == (size_t)numPolicyChannels * sizeof(float));
    assert(inputBuffers->singlePolicyResultElts == (size_t)numPolicyChannels * nnXLen * nnYLen);
    assert(inputBuffers->singlePolicyResultBytes == inputBuffers->singlePolicyResultElts * sizeof(float));
    assert(inputBuffers->singleValueResultElts == 3);
    assert(inputBuffers->singleValueResultBytes == 3 * sizeof(float));
    assert(inputBuffers->singleScoreValueResultElts == 6);
    assert(inputBuffers->singleScoreValueResultBytes == 6 * sizeof(float));
    assert(inputBuffers->singleOwnershipResultElts == (size_t)nnXLen * nnYLen);
    assert(inputBuffers->singleOwnershipResultBytes == inputBuffers->singleOwnershipResultElts * sizeof(float));

    for(int nIdx = 0; nIdx < batchSize; nIdx++) {
      if(
        inputBufs[nIdx]->boardXSizeForServer != nnXLen ||
        inputBufs[nIdx]->boardYSizeForServer != nnYLen
      ) {
        throw StringError("Transformer CUDA backend 仅支持与模型 pos_len 完全一致的棋盘尺寸");
      }

      float* rowSpatialInput = inputBuffers->userInputBuffer + (inputBuffers->singleInputElts * nIdx);
      float* rowGlobalInput = inputBuffers->userInputGlobalBuffer + (inputBuffers->singleInputGlobalElts * nIdx);
      const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
      const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
      std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);
      SymmetryHelpers::copyInputsWithSymmetry(
        rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, gpuHandle->inputsUseNHWC, inputBufs[nIdx]->symmetry
      );
    }

    TransformerBuffers* buffers = gpuHandle->transformerBuffers.get();
    ScratchBuffers* scratch = gpuHandle->scratch.get();
    CUDA_ERR("getOutput", cudaMemcpy(buffers->inputBuf, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes * batchSize, cudaMemcpyHostToDevice));
    CUDA_ERR("getOutput", cudaMemcpy(buffers->inputGlobalBuf, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes * batchSize, cudaMemcpyHostToDevice));

    gpuHandle->transformerModel->apply(
      gpuHandle->cudaHandles.get(),
      scratch,
      batchSize,
      buffers->inputBuf,
      buffers->inputGlobalBuf,
      buffers->policyPassBuf,
      buffers->policyBuf,
      buffers->valueBuf,
      buffers->scoreValueBuf,
      buffers->ownershipBuf,
      buffers->workspaceBuf,
      buffers->workspaceBytes
    );

    CUDA_ERR("getOutput", cudaMemcpy(inputBuffers->policyPassResults, buffers->policyPassBuf, inputBuffers->singlePolicyPassResultBytes * batchSize, cudaMemcpyDeviceToHost));
    CUDA_ERR("getOutput", cudaMemcpy(inputBuffers->policyResults, buffers->policyBuf, inputBuffers->singlePolicyResultBytes * batchSize, cudaMemcpyDeviceToHost));
    CUDA_ERR("getOutput", cudaMemcpy(inputBuffers->valueResults, buffers->valueBuf, inputBuffers->singleValueResultBytes * batchSize, cudaMemcpyDeviceToHost));
    CUDA_ERR("getOutput", cudaMemcpy(inputBuffers->scoreValueResults, buffers->scoreValueBuf, inputBuffers->singleScoreValueResultBytes * batchSize, cudaMemcpyDeviceToHost));
    CUDA_ERR("getOutput", cudaMemcpy(inputBuffers->ownershipResults, buffers->ownershipBuf, inputBuffers->singleOwnershipResultBytes * batchSize, cudaMemcpyDeviceToHost));

    assert(outputs.size() == batchSize);

    for(int row = 0; row < batchSize; row++) {
      NNOutput* output = outputs[row];
      assert(output->nnXLen == nnXLen);
      assert(output->nnYLen == nnYLen);
      float policyOptimism = (float)inputBufs[row]->policyOptimism;

      const float* policyPassSrcBuf = inputBuffers->policyPassResults + row * numPolicyChannels;
      const float* policySrcBuf = inputBuffers->policyResults + row * numPolicyChannels * nnXLen * nnYLen;
      float* policyProbs = output->policyProbs;
      for(int i = 0; i < nnXLen * nnYLen; i++) {
        float p = policySrcBuf[i * numPolicyChannels];
        float pOpt = policySrcBuf[i * numPolicyChannels + 1];
        policyProbsTmp[i] = p + (pOpt - p) * policyOptimism;
      }
      SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;

      output->whiteWinProb = inputBuffers->valueResults[row * 3];
      output->whiteLossProb = inputBuffers->valueResults[row * 3 + 1];
      output->whiteNoResultProb = inputBuffers->valueResults[row * 3 + 2];

      if(output->whiteOwnerMap != NULL) {
        const float* ownershipSrcBuf = inputBuffers->ownershipResults + row * nnXLen * nnYLen;
        SymmetryHelpers::copyOutputsWithSymmetry(ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      }

      output->whiteScoreMean = inputBuffers->scoreValueResults[row * 6];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * 6 + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * 6 + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * 6 + 3];
      output->shorttermWinlossError = inputBuffers->scoreValueResults[row * 6 + 4];
      output->shorttermScoreError = inputBuffers->scoreValueResults[row * 6 + 5];
    }
    return;
  }
  const int modelVersion = gpuHandle->model->modelVersion;

  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  const int numMetaFeatures = inputBuffers->singleInputMetaElts;
  assert(numSpatialFeatures == gpuHandle->model->numInputChannels);
  assert(numSpatialFeatures * nnXLen * nnYLen == inputBuffers->singleInputElts);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);
  const int numPolicyChannels = gpuHandle->model->numPolicyChannels;

  for(int nIdx = 0; nIdx<batchSize; nIdx++) {
    float* rowSpatialInput = inputBuffers->userInputBuffer + (inputBuffers->singleInputElts * nIdx);
    float* rowGlobalInput = inputBuffers->userInputGlobalBuffer + (inputBuffers->singleInputGlobalElts * nIdx);
    float* rowMetaInput = inputBuffers->userInputMetaBuffer + (inputBuffers->singleInputMetaElts * nIdx);

    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
    bool hasRowMeta = inputBufs[nIdx]->hasRowMeta;
    std::copy(rowGlobal,rowGlobal+numGlobalFeatures,rowGlobalInput);
    if(numMetaFeatures > 0) {
      testAssert(rowMeta != NULL);
      testAssert(hasRowMeta);
      std::copy(rowMeta,rowMeta+numMetaFeatures,rowMetaInput);
    }
    else {
      testAssert(!hasRowMeta);
    }
    SymmetryHelpers::copyInputsWithSymmetry(rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, gpuHandle->inputsUseNHWC, inputBufs[nIdx]->symmetry);
  }

  Buffers* buffers = gpuHandle->buffers.get();
  ScratchBuffers* scratch = gpuHandle->scratch.get();

  if(!gpuHandle->usingFP16) {
    assert(inputBuffers->userInputBufferBytes == buffers->inputBufBytes);
    assert(inputBuffers->userInputGlobalBufferBytes == buffers->inputGlobalBufBytes);
    assert(inputBuffers->userInputMetaBufferBytes == buffers->inputMetaBufBytes);
    assert(inputBuffers->policyPassResultBufferBytes == buffers->policyPassBufBytes);
    assert(inputBuffers->policyResultBufferBytes == buffers->policyBufBytes);
    assert(inputBuffers->valueResultBufferBytes == buffers->valueBufBytes);
    assert(inputBuffers->singleInputBytes == inputBuffers->singleInputElts*4);
    assert(inputBuffers->singleInputGlobalBytes == inputBuffers->singleInputGlobalElts*4);
    assert(inputBuffers->singleInputMetaBytes == inputBuffers->singleInputMetaElts*4);
    assert(inputBuffers->singlePolicyPassResultElts == numPolicyChannels);
    assert(inputBuffers->singlePolicyPassResultBytes == numPolicyChannels * sizeof(float));
    assert(inputBuffers->singlePolicyResultElts == numPolicyChannels*nnXLen*nnYLen);
    assert(inputBuffers->singlePolicyResultBytes == numPolicyChannels*nnXLen*nnYLen * sizeof(float));
    assert(inputBuffers->scoreValueResultBufferBytes == buffers->scoreValueBufBytes);
    assert(inputBuffers->ownershipResultBufferBytes == buffers->ownershipBufBytes);
    assert(inputBuffers->singleOwnershipResultElts == nnXLen*nnYLen);
    assert(inputBuffers->singleOwnershipResultBytes == nnXLen*nnYLen * sizeof(float));

    CUDA_ERR("getOutput",cudaMemcpy(buffers->inputBuf, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes*batchSize, cudaMemcpyHostToDevice));
    CUDA_ERR("getOutput",cudaMemcpy(buffers->inputGlobalBuf, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes*batchSize, cudaMemcpyHostToDevice));
    if(numMetaFeatures > 0) {
      CUDA_ERR("getOutput",cudaMemcpy(buffers->inputMetaBuf, inputBuffers->userInputMetaBuffer, inputBuffers->singleInputMetaBytes*batchSize, cudaMemcpyHostToDevice));
    }
  }
  else {
    assert(inputBuffers->userInputBufferBytes == buffers->inputBufBytesFloat);
    assert(inputBuffers->userInputGlobalBufferBytes == buffers->inputGlobalBufBytesFloat);
    assert(inputBuffers->userInputMetaBufferBytes == buffers->inputMetaBufBytesFloat);
    assert(inputBuffers->policyResultBufferBytes == buffers->policyBufBytes);
    assert(inputBuffers->valueResultBufferBytes == buffers->valueBufBytes);
    assert(inputBuffers->userInputBufferBytes == buffers->inputBufBytes*2);
    assert(inputBuffers->userInputGlobalBufferBytes == buffers->inputGlobalBufBytes*2);
    assert(inputBuffers->userInputMetaBufferBytes == buffers->inputMetaBufBytes*2);
    assert(inputBuffers->singleInputBytes == inputBuffers->singleInputElts*4);
    assert(inputBuffers->singleInputGlobalBytes == inputBuffers->singleInputGlobalElts*4);
    assert(inputBuffers->singleInputMetaBytes == inputBuffers->singleInputMetaElts*4);
    assert(inputBuffers->singlePolicyPassResultElts == numPolicyChannels);
    assert(inputBuffers->singlePolicyPassResultBytes == numPolicyChannels * sizeof(float));
    assert(inputBuffers->singlePolicyResultElts == numPolicyChannels*nnXLen*nnYLen);
    assert(inputBuffers->singlePolicyResultBytes == numPolicyChannels*nnXLen*nnYLen * sizeof(float));
    assert(inputBuffers->scoreValueResultBufferBytes == buffers->scoreValueBufBytes);
    assert(inputBuffers->ownershipResultBufferBytes == buffers->ownershipBufBytes);
    assert(inputBuffers->singleOwnershipResultElts == nnXLen*nnYLen);
    assert(inputBuffers->singleOwnershipResultBytes == nnXLen*nnYLen * sizeof(float));

    CUDA_ERR("getOutput",cudaMemcpy(buffers->inputBufFloat, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes*batchSize, cudaMemcpyHostToDevice));
    CUDA_ERR("getOutput",cudaMemcpy(buffers->inputGlobalBufFloat, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes*batchSize, cudaMemcpyHostToDevice));
    if(numMetaFeatures > 0) {
      CUDA_ERR("getOutput",cudaMemcpy(buffers->inputMetaBufFloat, inputBuffers->userInputMetaBuffer, inputBuffers->singleInputMetaBytes*batchSize, cudaMemcpyHostToDevice));
    }

    customCudaCopyToHalf((const float*)buffers->inputBufFloat,(half*)buffers->inputBuf,inputBuffers->singleInputElts*batchSize);
    CUDA_ERR("getOutput",cudaPeekAtLastError());
    customCudaCopyToHalf((const float*)buffers->inputGlobalBufFloat,(half*)buffers->inputGlobalBuf,inputBuffers->singleInputGlobalElts*batchSize);
    CUDA_ERR("getOutput",cudaPeekAtLastError());
    if(numMetaFeatures > 0) {
      customCudaCopyToHalf((const float*)buffers->inputMetaBufFloat,(half*)buffers->inputMetaBuf,inputBuffers->singleInputMetaElts*batchSize);
      CUDA_ERR("getOutput",cudaPeekAtLastError());
    }
  }

  gpuHandle->model->apply(
    gpuHandle->cudaHandles.get(),
    scratch,
    batchSize,
    gpuHandle->requireExactNNLen,

    buffers->inputBuf,
    buffers->inputGlobalBuf,
    buffers->inputMetaBuf,

    buffers->policyPassBuf,
    buffers->policyBuf,

    buffers->valueBuf,
    buffers->scoreValueBuf,
    buffers->ownershipBuf,

    buffers->workspaceBuf,
    buffers->workspaceBytes
  );

  CUDA_ERR("getOutput",cudaMemcpy(inputBuffers->policyPassResults, buffers->policyPassBuf, inputBuffers->singlePolicyPassResultBytes*batchSize, cudaMemcpyDeviceToHost));
  CUDA_ERR("getOutput",cudaMemcpy(inputBuffers->policyResults, buffers->policyBuf, inputBuffers->singlePolicyResultBytes*batchSize, cudaMemcpyDeviceToHost));
  CUDA_ERR("getOutput",cudaMemcpy(inputBuffers->valueResults, buffers->valueBuf, inputBuffers->singleValueResultBytes*batchSize, cudaMemcpyDeviceToHost));
  CUDA_ERR("getOutput",cudaMemcpy(inputBuffers->scoreValueResults, buffers->scoreValueBuf, inputBuffers->singleScoreValueResultBytes*batchSize, cudaMemcpyDeviceToHost));
  CUDA_ERR("getOutput",cudaMemcpy(inputBuffers->ownershipResults, buffers->ownershipBuf, inputBuffers->singleOwnershipResultBytes*batchSize, cudaMemcpyDeviceToHost));

  assert(outputs.size() == batchSize);

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);
    float policyOptimism = (float)inputBufs[row]->policyOptimism;

    const float* policyPassSrcBuf = inputBuffers->policyPassResults + row * numPolicyChannels;
    const float* policySrcBuf = inputBuffers->policyResults + row * numPolicyChannels * nnXLen * nnYLen;
    float* policyProbs = output->policyProbs;

    // These are in logits, the client does the postprocessing to turn them into
    // policy probabilities and white game outcome probabilities
    // Also we don't fill in the nnHash here either
    // Handle version >= 12 policy optimism
    if(numPolicyChannels == 2 || (numPolicyChannels == 4 && modelVersion >= 16)) {
       if(gpuHandle->usingNHWC) {
        for(int i = 0; i<nnXLen*nnYLen; i++) {
          float p = policySrcBuf[i*numPolicyChannels];
          float pOpt = policySrcBuf[i*numPolicyChannels+1];
          policyProbsTmp[i] = p + (pOpt-p) * policyOptimism;
        }
        SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[nnXLen*nnYLen] = policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
      }
      else {
        for(int i = 0; i<nnXLen*nnYLen; i++) {
          float p = policySrcBuf[i];
          float pOpt = policySrcBuf[i+nnXLen*nnYLen];
          policyProbsTmp[i] = p + (pOpt-p) * policyOptimism;
        }
        SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[nnXLen*nnYLen] = policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
      }
    }
    else {
      assert(numPolicyChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      policyProbs[nnXLen*nnYLen] = policyPassSrcBuf[0];
    }

    int numValueChannels = gpuHandle->model->numValueChannels;
    assert(numValueChannels == 3);
    output->whiteWinProb = inputBuffers->valueResults[row * numValueChannels];
    output->whiteLossProb = inputBuffers->valueResults[row * numValueChannels + 1];
    output->whiteNoResultProb = inputBuffers->valueResults[row * numValueChannels + 2];

    //As above, these are NOT actually from white's perspective, but rather the player to move.
    //As usual the client does the postprocessing.
    if(output->whiteOwnerMap != NULL) {
      const float* ownershipSrcBuf = inputBuffers->ownershipResults + row * nnXLen * nnYLen;
      assert(gpuHandle->model->numOwnershipChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }

    if(modelVersion >= 9) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 6);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 4];
      output->shorttermScoreError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 5];
    }
    else if(modelVersion >= 8) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 4);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(modelVersion >= 4) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 2);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(modelVersion >= 3) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 1);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      //Version 3 neural nets don't have any second moment output, implicitly already folding it in, so we just use the mean squared
      output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }

}

bool NeuralNet::getTransformerRawOutputs(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  TransformerRawOutputs& outputs
) {
  if(!gpuHandle->isTransformer)
    return false;

  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);
  const int batchSize = numBatchEltsFilled;
  const int nnXLen = gpuHandle->nnXLen;
  const int nnYLen = gpuHandle->nnYLen;
  const int modelVersion = gpuHandle->transformerModel->modelVersion;
  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  const int numPolicyChannels = gpuHandle->transformerModel->numPolicyChannels;

  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    if(
      inputBufs[nIdx]->boardXSizeForServer != nnXLen ||
      inputBufs[nIdx]->boardYSizeForServer != nnYLen
    ) {
      throw StringError("Transformer CUDA backend 仅支持与模型 pos_len 完全一致的棋盘尺寸");
    }

    float* rowSpatialInput = inputBuffers->userInputBuffer + (inputBuffers->singleInputElts * nIdx);
    float* rowGlobalInput = inputBuffers->userInputGlobalBuffer + (inputBuffers->singleInputGlobalElts * nIdx);
    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);
    SymmetryHelpers::copyInputsWithSymmetry(
      rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, gpuHandle->inputsUseNHWC, inputBufs[nIdx]->symmetry
    );
  }

  TransformerBuffers* buffers = gpuHandle->transformerBuffers.get();
  ScratchBuffers* scratch = gpuHandle->scratch.get();
  CUDA_ERR("getTransformerRawOutputs", cudaMemcpy(buffers->inputBuf, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes * batchSize, cudaMemcpyHostToDevice));
  CUDA_ERR("getTransformerRawOutputs", cudaMemcpy(buffers->inputGlobalBuf, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes * batchSize, cudaMemcpyHostToDevice));

  gpuHandle->transformerModel->apply(
    gpuHandle->cudaHandles.get(),
    scratch,
    batchSize,
    buffers->inputBuf,
    buffers->inputGlobalBuf,
    buffers->policyPassBuf,
    buffers->policyBuf,
    buffers->valueBuf,
    buffers->scoreValueBuf,
    buffers->ownershipBuf,
    buffers->workspaceBuf,
    buffers->workspaceBytes
  );

  outputs.batchSize = batchSize;
  outputs.nnXLen = nnXLen;
  outputs.nnYLen = nnYLen;
  outputs.numPolicyChannels = numPolicyChannels;
  outputs.numFullPolicyChannels = 0;
  outputs.numMiscChannels = 0;
  outputs.numMoreMiscChannels = 0;
  outputs.numScoringChannels = 0;
  outputs.numFuturePosChannels = 0;
  outputs.numSekiChannels = 0;
  outputs.scoreBeliefLen = 0;
  outputs.policyPass.resize((size_t)batchSize * numPolicyChannels);
  outputs.policy.resize((size_t)batchSize * numPolicyChannels * nnXLen * nnYLen);
  outputs.value.resize((size_t)batchSize * 3);
  outputs.scoreValue.resize((size_t)batchSize * 6);
  outputs.ownership.resize((size_t)batchSize * nnXLen * nnYLen);
  outputs.fullPolicyPass.clear();
  outputs.fullPolicy.clear();
  outputs.misc.clear();
  outputs.moreMisc.clear();
  outputs.scoring.clear();
  outputs.futurePos.clear();
  outputs.seki.clear();
  outputs.scoreBelief.clear();

  CUDA_ERR("getTransformerRawOutputs", cudaMemcpy(outputs.policyPass.data(), buffers->policyPassBuf, sizeof(float) * outputs.policyPass.size(), cudaMemcpyDeviceToHost));
  CUDA_ERR("getTransformerRawOutputs", cudaMemcpy(outputs.policy.data(), buffers->policyBuf, sizeof(float) * outputs.policy.size(), cudaMemcpyDeviceToHost));
  CUDA_ERR("getTransformerRawOutputs", cudaMemcpy(outputs.value.data(), buffers->valueBuf, sizeof(float) * outputs.value.size(), cudaMemcpyDeviceToHost));
  CUDA_ERR("getTransformerRawOutputs", cudaMemcpy(outputs.scoreValue.data(), buffers->scoreValueBuf, sizeof(float) * outputs.scoreValue.size(), cudaMemcpyDeviceToHost));
  CUDA_ERR("getTransformerRawOutputs", cudaMemcpy(outputs.ownership.data(), buffers->ownershipBuf, sizeof(float) * outputs.ownership.size(), cudaMemcpyDeviceToHost));

  if(gpuHandle->transformerModel->supportsFullRawOutputs()) {
    const TransformerModel& model = *gpuHandle->transformerModel;
    outputs.numFullPolicyChannels = model.numFullPolicyChannels;
    outputs.numMiscChannels = model.numMiscChannels;
    outputs.numMoreMiscChannels = model.numMoreMiscChannels;
    outputs.numScoringChannels = model.numScoringChannels;
    outputs.numFuturePosChannels = model.numFuturePosChannels;
    outputs.numSekiChannels = model.numSekiChannels;
    outputs.scoreBeliefLen = model.scoreBeliefLen;
    outputs.fullPolicyPass.resize((size_t)batchSize * outputs.numFullPolicyChannels);
    outputs.fullPolicy.resize((size_t)batchSize * outputs.numFullPolicyChannels * nnXLen * nnYLen);
    outputs.misc.resize((size_t)batchSize * outputs.numMiscChannels);
    outputs.moreMisc.resize((size_t)batchSize * outputs.numMoreMiscChannels);
    outputs.scoring.resize((size_t)batchSize * nnXLen * nnYLen * outputs.numScoringChannels);
    outputs.futurePos.resize((size_t)batchSize * nnXLen * nnYLen * outputs.numFuturePosChannels);
    outputs.seki.resize((size_t)batchSize * nnXLen * nnYLen * outputs.numSekiChannels);
    outputs.scoreBelief.resize((size_t)batchSize * outputs.scoreBeliefLen);

    float* fullPolicyPassBuf = nullptr;
    float* fullPolicyBuf = nullptr;
    float* miscBuf = nullptr;
    float* moreMiscBuf = nullptr;
    float* scoringBuf = nullptr;
    float* futurePosBuf = nullptr;
    float* sekiBuf = nullptr;
    float* scoreBeliefProjectBuf = nullptr;
    vector<float> scoreBeliefProject;

    auto freeTemp = [&]() {
      if(fullPolicyPassBuf != nullptr) cudaFree(fullPolicyPassBuf);
      if(fullPolicyBuf != nullptr) cudaFree(fullPolicyBuf);
      if(miscBuf != nullptr) cudaFree(miscBuf);
      if(moreMiscBuf != nullptr) cudaFree(moreMiscBuf);
      if(scoringBuf != nullptr) cudaFree(scoringBuf);
      if(futurePosBuf != nullptr) cudaFree(futurePosBuf);
      if(sekiBuf != nullptr) cudaFree(sekiBuf);
      if(scoreBeliefProjectBuf != nullptr) cudaFree(scoreBeliefProjectBuf);
    };

    try {
      CUDA_ERR("getTransformerRawOutputs", cudaMalloc(reinterpret_cast<void**>(&fullPolicyPassBuf), sizeof(float) * outputs.fullPolicyPass.size()));
      CUDA_ERR("getTransformerRawOutputs", cudaMalloc(reinterpret_cast<void**>(&fullPolicyBuf), sizeof(float) * outputs.fullPolicy.size()));
      CUDA_ERR("getTransformerRawOutputs", cudaMalloc(reinterpret_cast<void**>(&miscBuf), sizeof(float) * outputs.misc.size()));
      CUDA_ERR("getTransformerRawOutputs", cudaMalloc(reinterpret_cast<void**>(&moreMiscBuf), sizeof(float) * outputs.moreMisc.size()));
      CUDA_ERR("getTransformerRawOutputs", cudaMalloc(reinterpret_cast<void**>(&scoringBuf), sizeof(float) * outputs.scoring.size()));
      CUDA_ERR("getTransformerRawOutputs", cudaMalloc(reinterpret_cast<void**>(&futurePosBuf), sizeof(float) * outputs.futurePos.size()));
      CUDA_ERR("getTransformerRawOutputs", cudaMalloc(reinterpret_cast<void**>(&sekiBuf), sizeof(float) * outputs.seki.size()));
      CUDA_ERR(
        "getTransformerRawOutputs",
        cudaMalloc(reinterpret_cast<void**>(&scoreBeliefProjectBuf), sizeof(float) * (size_t)batchSize * model.scoreBeliefProjectSize)
      );

      model.applyFull(
        gpuHandle->cudaHandles.get(),
        scratch,
        batchSize,
        buffers->inputBuf,
        buffers->inputGlobalBuf,
        fullPolicyPassBuf,
        fullPolicyBuf,
        buffers->valueBuf,
        miscBuf,
        moreMiscBuf,
        buffers->ownershipBuf,
        scoringBuf,
        futurePosBuf,
        sekiBuf,
        scoreBeliefProjectBuf,
        buffers->workspaceBuf,
        buffers->workspaceBytes
      );

      CUDA_ERR("getTransformerRawOutputs", cudaMemcpy(outputs.fullPolicyPass.data(), fullPolicyPassBuf, sizeof(float) * outputs.fullPolicyPass.size(), cudaMemcpyDeviceToHost));
      CUDA_ERR("getTransformerRawOutputs", cudaMemcpy(outputs.fullPolicy.data(), fullPolicyBuf, sizeof(float) * outputs.fullPolicy.size(), cudaMemcpyDeviceToHost));
      CUDA_ERR("getTransformerRawOutputs", cudaMemcpy(outputs.value.data(), buffers->valueBuf, sizeof(float) * outputs.value.size(), cudaMemcpyDeviceToHost));
      CUDA_ERR("getTransformerRawOutputs", cudaMemcpy(outputs.misc.data(), miscBuf, sizeof(float) * outputs.misc.size(), cudaMemcpyDeviceToHost));
      CUDA_ERR("getTransformerRawOutputs", cudaMemcpy(outputs.moreMisc.data(), moreMiscBuf, sizeof(float) * outputs.moreMisc.size(), cudaMemcpyDeviceToHost));
      CUDA_ERR("getTransformerRawOutputs", cudaMemcpy(outputs.ownership.data(), buffers->ownershipBuf, sizeof(float) * outputs.ownership.size(), cudaMemcpyDeviceToHost));
      CUDA_ERR("getTransformerRawOutputs", cudaMemcpy(outputs.scoring.data(), scoringBuf, sizeof(float) * outputs.scoring.size(), cudaMemcpyDeviceToHost));
      CUDA_ERR("getTransformerRawOutputs", cudaMemcpy(outputs.futurePos.data(), futurePosBuf, sizeof(float) * outputs.futurePos.size(), cudaMemcpyDeviceToHost));
      CUDA_ERR("getTransformerRawOutputs", cudaMemcpy(outputs.seki.data(), sekiBuf, sizeof(float) * outputs.seki.size(), cudaMemcpyDeviceToHost));

      scoreBeliefProject.resize((size_t)batchSize * model.scoreBeliefProjectSize);
      CUDA_ERR("getTransformerRawOutputs", cudaMemcpy(scoreBeliefProject.data(), scoreBeliefProjectBuf, sizeof(float) * scoreBeliefProject.size(), cudaMemcpyDeviceToHost));
      for(int row = 0; row < batchSize; row++) {
        finalizeTransformerScoreBelief(
          model,
          inputBuffers->userInputGlobalBuffer + (size_t)row * numGlobalFeatures,
          vector<float>(
            scoreBeliefProject.begin() + (size_t)row * model.scoreBeliefProjectSize,
            scoreBeliefProject.begin() + (size_t)(row + 1) * model.scoreBeliefProjectSize
          ),
          outputs.scoreBelief.data() + (size_t)row * outputs.scoreBeliefLen
        );
      }
      freeTemp();
    }
    catch(...) {
      freeTemp();
      throw;
    }
  }
  return true;
}

//TESTING ----------------------------------------------------------------------------------


bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  vector<float>& outputBuffer
) {
  cudaDeviceSynchronize();
  CudaHandles* cudaHandles = CudaHandles::cudaHandlesTesting();

  size_t numInputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->inChannels;
  size_t numOutputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->outChannels;
  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateConv: unexpected input buffer size");

  void* deviceInput;
  void* deviceOutput;
  CudaUtils::mallocAndCopyToDevice("deviceInput", inputBuffer.data(), numInputFloats, deviceInput, useFP16);
  CudaUtils::mallocOnDevice("deviceOutput", numOutputFloats, deviceOutput, useFP16);

  int maxBatchSize = desiredBatchSize;

  CudnnManager* manager = new CudnnManager("manager",maxBatchSize,nnXLen,nnYLen);
  ConvLayer* convLayer = new ConvLayer(cudaHandles,manager,desc,useFP16,useNHWC);

  size_t workspaceBytes =
    convLayer->requiredWorkspaceBytes(cudaHandles,desiredBatchSize);
  void* deviceWorkspace;
  CUDA_ERR("deviceWorkspace",cudaMalloc(&deviceWorkspace, workspaceBytes));


  bool accumulate = false;
  convLayer->apply(
    cudaHandles,
    desiredBatchSize,
    accumulate,
    deviceInput,
    deviceOutput,
    deviceWorkspace,
    workspaceBytes
  );

  outputBuffer.resize(numOutputFloats);
  CudaUtils::expensiveCopyFromDevice("copyResultsToHost", outputBuffer.data(), numOutputFloats, deviceOutput, useFP16);

  cudaFree(deviceWorkspace);

  delete convLayer;
  delete manager;
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  delete cudaHandles;

  return true;
}


bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  cudaDeviceSynchronize();
  CudaHandles* cudaHandles = CudaHandles::cudaHandlesTesting();

  size_t numInputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->numChannels;
  size_t numMaskFloats = (size_t)desiredBatchSize * nnXLen * nnYLen;
  size_t numOutputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->numChannels;
  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateBatchNorm: unexpected input buffer size");
  if(numMaskFloats != maskBuffer.size())
    throw StringError("testEvaluateBatchNorm: unexpected mask buffer size");

  ActivationLayerDesc actDesc;
  actDesc.activation = ACTIVATION_IDENTITY;

  void* deviceInput;
  void* deviceMask;
  void* deviceOutput;
  CudaUtils::mallocAndCopyToDevice("deviceInput", inputBuffer.data(), numInputFloats, deviceInput, useFP16);
  CudaUtils::mallocAndCopyToDevice("deviceMask", maskBuffer.data(), numMaskFloats, deviceMask, useFP16);
  CudaUtils::mallocOnDevice("deviceOutput", numOutputFloats, deviceOutput, useFP16);

  BatchNormLayer* batchNormLayer = new BatchNormLayer(cudaHandles,desc,&actDesc,nnXLen,nnYLen,useFP16,useNHWC);

  batchNormLayer->apply(
    cudaHandles,
    desiredBatchSize,
    deviceInput,
    deviceMask,
    deviceOutput
  );

  outputBuffer.resize(numOutputFloats);
  CudaUtils::expensiveCopyFromDevice("copyResultsToHost", outputBuffer.data(), numOutputFloats, deviceOutput, useFP16);

  delete batchNormLayer;

  cudaFree(deviceInput);
  cudaFree(deviceMask);
  cudaFree(deviceOutput);
  delete cudaHandles;

  return true;
}


bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  cudaDeviceSynchronize();
  CudaHandles* cudaHandles = CudaHandles::cudaHandlesTesting();

  size_t numInputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->preBN.numChannels;
  size_t numMaskFloats = (size_t)desiredBatchSize * nnXLen * nnYLen;
  size_t numOutputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->finalConv.outChannels;
  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateResidualBlock: unexpected input buffer size");
  if(numMaskFloats != maskBuffer.size())
    throw StringError("testEvaluateResidualBlock: unexpected mask buffer size");

  ScratchBuffers* scratch = new ScratchBuffers(desiredBatchSize, nnXLen, nnYLen, useFP16);

  void* deviceInput;
  void* deviceMask;
  void* deviceScratch;
  CudaUtils::mallocAndCopyToDevice("deviceInput", inputBuffer.data(), numInputFloats, deviceInput, useFP16);
  CudaUtils::mallocAndCopyToDevice("deviceMask", maskBuffer.data(), numMaskFloats, deviceMask, useFP16);
  CudaUtils::mallocOnDevice("deviceScratch", numInputFloats, deviceScratch, useFP16);

  int maxBatchSize = desiredBatchSize;

  CudnnManager* manager = new CudnnManager("manager",maxBatchSize,nnXLen,nnYLen);
  ResidualBlock* residualBlock = new ResidualBlock(cudaHandles,manager,desc,nnXLen,nnYLen,useFP16,useNHWC);

  size_t workspaceBytes =
    residualBlock->requiredWorkspaceBytes(cudaHandles,desiredBatchSize);
  void* deviceWorkspace;
  CUDA_ERR("deviceWorkspace",cudaMalloc(&deviceWorkspace, workspaceBytes));

  residualBlock->apply(
    cudaHandles,
    scratch,
    desiredBatchSize,
    deviceInput,
    deviceScratch,
    deviceMask,
    deviceWorkspace,
    workspaceBytes
  );

  outputBuffer.resize(numOutputFloats);
  CudaUtils::expensiveCopyFromDevice("copyResultsToHost", outputBuffer.data(), numOutputFloats, deviceInput, useFP16);

  cudaFree(deviceWorkspace);

  delete residualBlock;
  delete manager;
  cudaFree(deviceInput);
  cudaFree(deviceMask);
  cudaFree(deviceScratch);
  delete scratch;
  delete cudaHandles;

  return true;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  cudaDeviceSynchronize();
  CudaHandles* cudaHandles = CudaHandles::cudaHandlesTesting();

  size_t numInputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->preBN.numChannels;
  size_t numMaskFloats = (size_t)desiredBatchSize * nnXLen * nnYLen;
  size_t numMaskSumFloats = (size_t)desiredBatchSize;
  size_t numOutputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->finalConv.outChannels;

  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateGlobalPoolingResidualBlock: unexpected input buffer size");
  if(numMaskFloats != maskBuffer.size())
    throw StringError("testEvaluateGlobalPoolingResidualBlock: unexpected mask buffer size");

  ScratchBuffers* scratch = new ScratchBuffers(desiredBatchSize, nnXLen, nnYLen, useFP16);

  void* deviceInput;
  void* deviceMask;
  float* deviceMaskFloatOrig;
  float* deviceMaskFloat;
  float* deviceMaskSum;
  void* deviceScratch;

  CudaUtils::mallocAndCopyToDevice("deviceInput", inputBuffer.data(), numInputFloats, deviceInput, useFP16);
  CudaUtils::mallocAndCopyToDevice("deviceMask", maskBuffer.data(), numMaskFloats, deviceMask, useFP16);
  CUDA_ERR("deviceMaskFloat",cudaMalloc(reinterpret_cast<void**>(&deviceMaskFloat), numMaskFloats * sizeof(float)));
  CUDA_ERR("deviceMaskSum",cudaMalloc(reinterpret_cast<void**>(&deviceMaskSum), numMaskSumFloats * sizeof(float)));
  deviceMaskFloatOrig = deviceMaskFloat;
  CudaUtils::mallocOnDevice("deviceScratch", numInputFloats, deviceScratch, useFP16);

  fillMaskFloatBufAndMaskSumBuf(deviceMask, deviceMaskFloat, deviceMaskSum, useFP16, desiredBatchSize, nnXLen, nnYLen);

  int maxBatchSize = desiredBatchSize;

  CudnnManager* manager = new CudnnManager("manager",maxBatchSize,nnXLen,nnYLen);
  GlobalPoolingResidualBlock* residualBlock = new GlobalPoolingResidualBlock(
    cudaHandles,manager,desc,nnXLen,nnYLen,useFP16,useNHWC
  );

  size_t workspaceBytes =
    residualBlock->requiredWorkspaceBytes(
      cudaHandles,desiredBatchSize
    );

  void* deviceWorkspace;
  CUDA_ERR("deviceWorkspace",cudaMalloc(&deviceWorkspace, workspaceBytes));

  residualBlock->apply(
    cudaHandles,
    scratch,
    desiredBatchSize,
    deviceInput,
    deviceScratch,
    deviceMask,
    deviceMaskSum,
    deviceWorkspace,
    workspaceBytes
  );

  outputBuffer.resize(numOutputFloats);
  CudaUtils::expensiveCopyFromDevice("copyResultsToHost", outputBuffer.data(), numOutputFloats, deviceInput, useFP16);

  cudaFree(deviceWorkspace);

  delete residualBlock;
  delete manager;

  cudaFree(deviceInput);
  cudaFree(deviceMask);
  cudaFree(deviceMaskFloatOrig);
  cudaFree(deviceMaskSum);
  cudaFree(deviceScratch);
  delete scratch;
  delete cudaHandles;

  return true;
}


#endif  // USE_CUDA_BACKEND
