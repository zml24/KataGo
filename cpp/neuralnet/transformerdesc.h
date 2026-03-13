#ifndef NEURALNET_TRANSFORMERDESC_H_
#define NEURALNET_TRANSFORMERDESC_H_

#include <string>
#include <vector>

#include "../neuralnet/desc.h"

struct TransformerBlockDesc {
  std::vector<float> norm1Weight;
  std::vector<float> qWeight;
  std::vector<float> kWeight;
  std::vector<float> vWeight;
  std::vector<float> outWeight;
  std::vector<float> norm2Weight;
  std::vector<float> ffnW1Weight;
  std::vector<float> ffnWGateWeight;
  std::vector<float> ffnW2Weight;
};

struct TransformerModelDesc {
  int formatVersion;
  std::string name;
  std::string sha256;
  int modelVersion;
  int posLen;
  int hiddenSize;
  int numLayers;
  int numHeads;
  int headDim;
  int ffnDim;
  int numInputChannels;
  int numInputGlobalChannels;
  int stemKernelSize;
  int scoreMode;
  int numScoreBeliefs;
  int scoreBeliefLen;
  bool hasPosEmbed;

  ModelPostProcessParams postProcessParams;

  std::vector<float> stemConvWeight;
  std::vector<float> stemGlobalWeight;
  std::vector<float> posEmbed;
  std::vector<float> ropeCos;
  std::vector<float> ropeSin;
  std::vector<TransformerBlockDesc> blocks;
  std::vector<float> finalNormWeight;
  std::vector<float> policyBoardWeight;
  std::vector<float> policyPassWeight;
  std::vector<float> policyBoardFullWeight;
  std::vector<float> policyPassFullWeight;
  std::vector<float> valueWeight;
  std::vector<float> miscWeight;
  std::vector<float> moreMiscWeight;
  std::vector<float> scoreValueWeight;
  std::vector<float> ownershipWeight;
  std::vector<float> scoringWeight;
  std::vector<float> futurePosWeight;
  std::vector<float> sekiWeight;
  std::vector<float> scoreBeliefSimpleWeight;
  std::vector<float> scoreBeliefMixWeight;
  std::vector<float> scoreBeliefS2OffWeight;
  std::vector<float> scoreBeliefS2ParWeight;

  TransformerModelDesc();
  ~TransformerModelDesc();
  TransformerModelDesc(TransformerModelDesc&& other) = default;
  TransformerModelDesc& operator=(TransformerModelDesc&& other) = default;

  TransformerModelDesc(const TransformerModelDesc&) = delete;
  TransformerModelDesc& operator=(const TransformerModelDesc&) = delete;

  static bool tryLoadFromFileMaybeGZipped(
    const std::string& fileName,
    const std::string& expectedSha256,
    TransformerModelDesc& descBuf
  );
};

#endif  // NEURALNET_TRANSFORMERDESC_H_
