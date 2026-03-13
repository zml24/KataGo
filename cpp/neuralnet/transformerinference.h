#ifndef NEURALNET_TRANSFORMERINFERENCE_H_
#define NEURALNET_TRANSFORMERINFERENCE_H_

#include "../neuralnet/transformerdesc.h"

class TransformerInferenceEngine {
 public:
  explicit TransformerInferenceEngine(const TransformerModelDesc* desc);
  ~TransformerInferenceEngine();

  TransformerInferenceEngine() = delete;
  TransformerInferenceEngine(const TransformerInferenceEngine&) = delete;
  TransformerInferenceEngine& operator=(const TransformerInferenceEngine&) = delete;

  void apply(
    int batchSize,
    const float* spatialInputNHWC,
    const float* globalInput,
    float* policyPassOut,
    float* policyOut,
    float* valueOut,
    float* scoreValueOut,
    float* ownershipOut
  ) const;

  void applyFull(
    int batchSize,
    const float* spatialInputNHWC,
    const float* globalInput,
    float* policyPassFullOut,
    float* policyFullOut,
    float* valueOut,
    float* miscOut,
    float* moreMiscOut,
    float* ownershipOut,
    float* scoringOut,
    float* futurePosOut,
    float* sekiOut,
    float* scoreBeliefOut
  ) const;

  int getModelVersion() const;
  int getPosLen() const;
  int getNumInputChannels() const;
  int getNumInputGlobalChannels() const;
  int getNumPolicyChannels() const;
  int getNumFullPolicyChannels() const;
  int getScoreBeliefLen() const;
  bool supportsFullRawOutputs() const;

 private:
  const TransformerModelDesc* desc;
};

#endif  // NEURALNET_TRANSFORMERINFERENCE_H_
