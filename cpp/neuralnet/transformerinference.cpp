#include "../neuralnet/transformerinference.h"

#include <algorithm>
#include <cmath>

#include "../core/global.h"

using namespace std;

namespace {

static constexpr float RMS_EPS = 1e-6f;

static inline size_t idx2(int row, int col, int cols) {
  return (size_t)row * cols + col;
}

static void rmsNorm(const vector<float>& in, int rows, int cols, const vector<float>& weight, vector<float>& out) {
  out.resize((size_t)rows * cols);
  for(int r = 0; r < rows; r++) {
    double sqSum = 0.0;
    for(int c = 0; c < cols; c++) {
      float x = in[idx2(r,c,cols)];
      sqSum += (double)x * (double)x;
    }
    float scale = (float)(1.0 / std::sqrt(sqSum / cols + RMS_EPS));
    for(int c = 0; c < cols; c++)
      out[idx2(r,c,cols)] = in[idx2(r,c,cols)] * scale * weight[c];
  }
}

static void matmul(const vector<float>& a, int aRows, int aCols, const vector<float>& b, int bCols, vector<float>& out) {
  out.resize((size_t)aRows * bCols);
  for(int r = 0; r < aRows; r++) {
    for(int c = 0; c < bCols; c++) {
      double sum = 0.0;
      for(int k = 0; k < aCols; k++)
        sum += (double)a[idx2(r,k,aCols)] * (double)b[idx2(k,c,bCols)];
      out[idx2(r,c,bCols)] = (float)sum;
    }
  }
}

static void addInplace(vector<float>& dst, const vector<float>& src) {
  assert(dst.size() == src.size());
  for(size_t i = 0; i < dst.size(); i++)
    dst[i] += src[i];
}

static void applyRoPE(const vector<float>& cos, const vector<float>& sin, int tokenIdx, float* vec, int dim) {
  const int half = dim / 2;
  const float* cosRow = cos.data() + (size_t)tokenIdx * dim;
  const float* sinRow = sin.data() + (size_t)tokenIdx * dim;
  vector<float> rotated(dim);
  for(int i = 0; i < half; i++) {
    rotated[i] = -vec[i + half];
    rotated[i + half] = vec[i];
  }
  for(int i = 0; i < dim; i++)
    vec[i] = vec[i] * cosRow[i] + rotated[i] * sinRow[i];
}

static inline float silu(float x) {
  return x / (1.0f + std::exp(-x));
}

static vector<float> meanRows(const vector<float>& in, int rows, int cols) {
  vector<float> out(cols, 0.0f);
  if(rows == 0)
    return out;
  for(int r = 0; r < rows; r++) {
    for(int c = 0; c < cols; c++)
      out[c] += in[idx2(r,c,cols)];
  }
  for(int c = 0; c < cols; c++)
    out[c] = (float)((double)out[c] / (double)rows);
  return out;
}

static void softmaxRowsInplace(vector<float>& scores, int rows, int cols) {
  for(int r = 0; r < rows; r++) {
    float maxVal = scores[idx2(r,0,cols)];
    for(int c = 1; c < cols; c++)
      maxVal = std::max(maxVal, scores[idx2(r,c,cols)]);
    double sum = 0.0;
    for(int c = 0; c < cols; c++) {
      float v = (float)std::exp((double)scores[idx2(r,c,cols)] - (double)maxVal);
      scores[idx2(r,c,cols)] = v;
      sum += v;
    }
    float invSum = (float)(1.0 / sum);
    for(int c = 0; c < cols; c++)
      scores[idx2(r,c,cols)] *= invSum;
  }
}

static void runBlock(
  const TransformerModelDesc& desc,
  const TransformerBlockDesc& block,
  vector<float>& x
) {
  const int seqLen = desc.posLen * desc.posLen;
  const int hidden = desc.hiddenSize;
  const int numHeads = desc.numHeads;
  const int headDim = desc.headDim;
  const float invSqrtHeadDim = 1.0f / std::sqrt((float)headDim);

  vector<float> xNorm;
  rmsNorm(x, seqLen, hidden, block.norm1Weight, xNorm);

  vector<float> q;
  vector<float> k;
  vector<float> v;
  vector<float> attnOut((size_t)seqLen * hidden, 0.0f);
  matmul(xNorm, seqLen, hidden, block.qWeight, hidden, q);
  matmul(xNorm, seqLen, hidden, block.kWeight, hidden, k);
  matmul(xNorm, seqLen, hidden, block.vWeight, hidden, v);

  for(int head = 0; head < numHeads; head++) {
    const int offset = head * headDim;
    for(int t = 0; t < seqLen; t++) {
      applyRoPE(desc.ropeCos, desc.ropeSin, t, q.data() + idx2(t, offset, hidden), headDim);
      applyRoPE(desc.ropeCos, desc.ropeSin, t, k.data() + idx2(t, offset, hidden), headDim);
    }

    vector<float> scores((size_t)seqLen * seqLen, 0.0f);
    for(int i = 0; i < seqLen; i++) {
      for(int j = 0; j < seqLen; j++) {
        double dot = 0.0;
        const float* qRow = q.data() + idx2(i, offset, hidden);
        const float* kRow = k.data() + idx2(j, offset, hidden);
        for(int d = 0; d < headDim; d++)
          dot += (double)qRow[d] * (double)kRow[d];
        scores[idx2(i,j,seqLen)] = (float)(dot * invSqrtHeadDim);
      }
    }
    softmaxRowsInplace(scores, seqLen, seqLen);
    for(int i = 0; i < seqLen; i++) {
      float* outRow = attnOut.data() + idx2(i, offset, hidden);
      for(int j = 0; j < seqLen; j++) {
        float w = scores[idx2(i,j,seqLen)];
        const float* vRow = v.data() + idx2(j, offset, hidden);
        for(int d = 0; d < headDim; d++)
          outRow[d] += w * vRow[d];
      }
    }
  }

  vector<float> projOut;
  matmul(attnOut, seqLen, hidden, block.outWeight, hidden, projOut);
  addInplace(x, projOut);

  rmsNorm(x, seqLen, hidden, block.norm2Weight, xNorm);
  vector<float> ffn1;
  vector<float> ffnGate;
  matmul(xNorm, seqLen, hidden, block.ffnW1Weight, desc.ffnDim, ffn1);
  matmul(xNorm, seqLen, hidden, block.ffnWGateWeight, desc.ffnDim, ffnGate);
  for(size_t i = 0; i < ffn1.size(); i++)
    ffn1[i] = silu(ffn1[i]) * ffnGate[i];
  vector<float> ffn2;
  matmul(ffn1, seqLen, desc.ffnDim, block.ffnW2Weight, hidden, ffn2);
  addInplace(x, ffn2);
}

static void matmulVec(const vector<float>& vec, const vector<float>& weight, int inSize, int outSize, float* out) {
  for(int c = 0; c < outSize; c++) {
    double sum = 0.0;
    for(int i = 0; i < inSize; i++)
      sum += (double)vec[i] * (double)weight[(size_t)i * outSize + c];
    out[c] = (float)sum;
  }
}

static void computeTrunk(
  const TransformerModelDesc& desc,
  const float* spatialInputNHWC,
  const float* globalInput,
  vector<float>& xFinal,
  vector<float>& pooled
) {
  const int posLen = desc.posLen;
  const int seqLen = posLen * posLen;
  const int hidden = desc.hiddenSize;
  const int inChannels = desc.numInputChannels;
  const int kernel = desc.stemKernelSize;
  const int padding = kernel / 2;

  vector<float> x((size_t)seqLen * hidden, 0.0f);

  for(int y = 0; y < posLen; y++) {
    for(int xx = 0; xx < posLen; xx++) {
      const int tokenIdx = y * posLen + xx;
      for(int oc = 0; oc < hidden; oc++) {
        double sum = 0.0;
        for(int ic = 0; ic < inChannels; ic++) {
          for(int ky = 0; ky < kernel; ky++) {
            for(int kx = 0; kx < kernel; kx++) {
              int iy = y + ky - padding;
              int ix = xx + kx - padding;
              if(iy < 0 || iy >= posLen || ix < 0 || ix >= posLen)
                continue;
              size_t inputIdx = ((size_t)iy * posLen + ix) * inChannels + ic;
              size_t weightIdx = ((((size_t)oc * inChannels + ic) * kernel + ky) * kernel + kx);
              sum += (double)spatialInputNHWC[inputIdx] * (double)desc.stemConvWeight[weightIdx];
            }
          }
        }
        x[idx2(tokenIdx, oc, hidden)] = (float)sum;
      }
    }
  }

  for(int oc = 0; oc < hidden; oc++) {
    double sum = 0.0;
    for(int ic = 0; ic < desc.numInputGlobalChannels; ic++)
      sum += (double)globalInput[ic] * (double)desc.stemGlobalWeight[(size_t)ic * hidden + oc];
    float globalValue = (float)sum;
    for(int tokenIdx = 0; tokenIdx < seqLen; tokenIdx++)
      x[idx2(tokenIdx, oc, hidden)] += globalValue;
  }

  if(desc.hasPosEmbed) {
    for(int tokenIdx = 0; tokenIdx < seqLen; tokenIdx++) {
      const float* pos = desc.posEmbed.data() + (size_t)tokenIdx * hidden;
      for(int c = 0; c < hidden; c++)
        x[idx2(tokenIdx, c, hidden)] += pos[c];
    }
  }

  for(const TransformerBlockDesc& block : desc.blocks)
    runBlock(desc, block, x);

  rmsNorm(x, seqLen, hidden, desc.finalNormWeight, xFinal);
  pooled = meanRows(xFinal, seqLen, hidden);
}

static void logSoftmax1DInplace(vector<float>& values) {
  if(values.empty())
    return;
  float maxVal = values[0];
  for(size_t i = 1; i < values.size(); i++)
    maxVal = std::max(maxVal, values[i]);
  double sum = 0.0;
  for(size_t i = 0; i < values.size(); i++)
    sum += std::exp((double)values[i] - (double)maxVal);
  float logSum = (float)((double)maxVal + std::log(sum));
  for(size_t i = 0; i < values.size(); i++)
    values[i] -= logSum;
}

static void computeScoreBelief(
  const TransformerModelDesc& desc,
  const vector<float>& pooled,
  float scoreParity,
  float* scoreBeliefOut
) {
  if(desc.scoreMode < 0 || desc.scoreBeliefLen <= 0)
    throw StringError("Transformer full raw outputs unavailable: missing score belief metadata");

  if(desc.scoreMode == 0) {
    vector<float> logits((size_t)desc.scoreBeliefLen);
    matmulVec(pooled, desc.scoreBeliefSimpleWeight, desc.hiddenSize, desc.scoreBeliefLen, logits.data());
    logSoftmax1DInplace(logits);
    std::copy(logits.begin(), logits.end(), scoreBeliefOut);
    return;
  }

  const int len = desc.scoreBeliefLen;
  const int numBeliefs = desc.numScoreBeliefs;
  vector<float> proj((size_t)len * numBeliefs + numBeliefs);
  matmulVec(
    pooled,
    desc.scoreBeliefMixWeight,
    desc.hiddenSize,
    len * numBeliefs + numBeliefs,
    proj.data()
  );

  vector<float> belief(proj.begin(), proj.begin() + (size_t)len * numBeliefs);
  vector<float> mixLogits(proj.begin() + (size_t)len * numBeliefs, proj.end());

  if(desc.scoreMode == 2) {
    const int mid = len / 2;
    for(int i = 0; i < len; i++) {
      int diff = i - mid;
      int parityBit = ((diff % 2) + 2) % 2;
      float offsetTerm = 0.05f * ((float)diff + 0.5f);
      float parityTerm = (0.5f - (float)parityBit) * scoreParity;
      for(int j = 0; j < numBeliefs; j++) {
        belief[(size_t)i * numBeliefs + j] +=
          offsetTerm * desc.scoreBeliefS2OffWeight[j] +
          parityTerm * desc.scoreBeliefS2ParWeight[j];
      }
    }
  }

  for(int j = 0; j < numBeliefs; j++) {
    float maxVal = belief[j];
    for(int i = 1; i < len; i++)
      maxVal = std::max(maxVal, belief[(size_t)i * numBeliefs + j]);
    double sum = 0.0;
    for(int i = 0; i < len; i++)
      sum += std::exp((double)belief[(size_t)i * numBeliefs + j] - (double)maxVal);
    float logSum = (float)((double)maxVal + std::log(sum));
    for(int i = 0; i < len; i++)
      belief[(size_t)i * numBeliefs + j] -= logSum;
  }

  logSoftmax1DInplace(mixLogits);
  for(int i = 0; i < len; i++) {
    float maxVal = belief[(size_t)i * numBeliefs] + mixLogits[0];
    for(int j = 1; j < numBeliefs; j++)
      maxVal = std::max(maxVal, belief[(size_t)i * numBeliefs + j] + mixLogits[j]);
    double sum = 0.0;
    for(int j = 0; j < numBeliefs; j++)
      sum += std::exp((double)belief[(size_t)i * numBeliefs + j] + (double)mixLogits[j] - (double)maxVal);
    scoreBeliefOut[i] = (float)((double)maxVal + std::log(sum));
  }
}

static void runSingle(
  const TransformerModelDesc& desc,
  const float* spatialInputNHWC,
  const float* globalInput,
  float* policyPassOut,
  float* policyOut,
  float* valueOut,
  float* scoreValueOut,
  float* ownershipOut
) {
  const int seqLen = desc.posLen * desc.posLen;
  vector<float> xFinal;
  vector<float> pooled;
  computeTrunk(desc, spatialInputNHWC, globalInput, xFinal, pooled);

  vector<float> boardPolicy;
  vector<float> ownership;
  matmul(xFinal, seqLen, desc.hiddenSize, desc.policyBoardWeight, 2, boardPolicy);
  matmul(xFinal, seqLen, desc.hiddenSize, desc.ownershipWeight, 1, ownership);
  for(int tokenIdx = 0; tokenIdx < seqLen; tokenIdx++) {
    for(int c = 0; c < 2; c++)
      policyOut[tokenIdx * 2 + c] = boardPolicy[idx2(tokenIdx, c, 2)];
    ownershipOut[tokenIdx] = ownership[tokenIdx];
  }
  matmulVec(pooled, desc.policyPassWeight, desc.hiddenSize, 2, policyPassOut);
  matmulVec(pooled, desc.valueWeight, desc.hiddenSize, 3, valueOut);
  matmulVec(pooled, desc.scoreValueWeight, desc.hiddenSize, 6, scoreValueOut);
}

static void runSingleFull(
  const TransformerModelDesc& desc,
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
) {
  if(desc.policyBoardFullWeight.empty() || desc.policyPassFullWeight.empty())
    throw StringError("Transformer full raw outputs unavailable: model file lacks full head weights");

  const int seqLen = desc.posLen * desc.posLen;
  vector<float> xFinal;
  vector<float> pooled;
  computeTrunk(desc, spatialInputNHWC, globalInput, xFinal, pooled);

  vector<float> fullBoardPolicy;
  vector<float> ownership;
  vector<float> scoring;
  vector<float> futurePos;
  vector<float> seki;
  matmul(xFinal, seqLen, desc.hiddenSize, desc.policyBoardFullWeight, 6, fullBoardPolicy);
  matmul(xFinal, seqLen, desc.hiddenSize, desc.ownershipWeight, 1, ownership);
  matmul(xFinal, seqLen, desc.hiddenSize, desc.scoringWeight, 1, scoring);
  matmul(xFinal, seqLen, desc.hiddenSize, desc.futurePosWeight, 2, futurePos);
  matmul(xFinal, seqLen, desc.hiddenSize, desc.sekiWeight, 4, seki);

  for(int tokenIdx = 0; tokenIdx < seqLen; tokenIdx++) {
    for(int c = 0; c < 6; c++)
      policyFullOut[tokenIdx * 6 + c] = fullBoardPolicy[idx2(tokenIdx, c, 6)];
    ownershipOut[tokenIdx] = ownership[tokenIdx];
    scoringOut[tokenIdx] = scoring[tokenIdx];
    for(int c = 0; c < 2; c++)
      futurePosOut[tokenIdx * 2 + c] = futurePos[idx2(tokenIdx, c, 2)];
    for(int c = 0; c < 4; c++)
      sekiOut[tokenIdx * 4 + c] = seki[idx2(tokenIdx, c, 4)];
  }

  matmulVec(pooled, desc.policyPassFullWeight, desc.hiddenSize, 6, policyPassFullOut);
  matmulVec(pooled, desc.valueWeight, desc.hiddenSize, 3, valueOut);
  matmulVec(pooled, desc.miscWeight, desc.hiddenSize, 10, miscOut);
  matmulVec(pooled, desc.moreMiscWeight, desc.hiddenSize, 8, moreMiscOut);
  computeScoreBelief(desc, pooled, globalInput[desc.numInputGlobalChannels - 1], scoreBeliefOut);
}

}

TransformerInferenceEngine::TransformerInferenceEngine(const TransformerModelDesc* d)
  : desc(d)
{
  if(desc == nullptr)
    throw StringError("TransformerInferenceEngine received a null model desc");
  if(desc->headDim % 2 != 0)
    throw StringError("TransformerInferenceEngine requires an even head dimension for RoPE");
}

TransformerInferenceEngine::~TransformerInferenceEngine() {}

void TransformerInferenceEngine::apply(
  int batchSize,
  const float* spatialInputNHWC,
  const float* globalInput,
  float* policyPassOut,
  float* policyOut,
  float* valueOut,
  float* scoreValueOut,
  float* ownershipOut
) const {
  const size_t singleSpatial = (size_t)desc->posLen * desc->posLen * desc->numInputChannels;
  const size_t singleGlobal = (size_t)desc->numInputGlobalChannels;
  const size_t singlePolicyPass = 2;
  const size_t singlePolicy = (size_t)desc->posLen * desc->posLen * 2;
  const size_t singleValue = 3;
  const size_t singleScore = 6;
  const size_t singleOwnership = (size_t)desc->posLen * desc->posLen;

  for(int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
    runSingle(
      *desc,
      spatialInputNHWC + batchIdx * singleSpatial,
      globalInput + batchIdx * singleGlobal,
      policyPassOut + batchIdx * singlePolicyPass,
      policyOut + batchIdx * singlePolicy,
      valueOut + batchIdx * singleValue,
      scoreValueOut + batchIdx * singleScore,
      ownershipOut + batchIdx * singleOwnership
    );
  }
}

void TransformerInferenceEngine::applyFull(
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
) const {
  const size_t singleSpatial = (size_t)desc->posLen * desc->posLen * desc->numInputChannels;
  const size_t singleGlobal = (size_t)desc->numInputGlobalChannels;
  const size_t singlePolicyPass = 6;
  const size_t singlePolicy = (size_t)desc->posLen * desc->posLen * 6;
  const size_t singleValue = 3;
  const size_t singleMisc = 10;
  const size_t singleMoreMisc = 8;
  const size_t singleOwnership = (size_t)desc->posLen * desc->posLen;
  const size_t singleScoring = (size_t)desc->posLen * desc->posLen;
  const size_t singleFuturePos = (size_t)desc->posLen * desc->posLen * 2;
  const size_t singleSeki = (size_t)desc->posLen * desc->posLen * 4;
  const size_t singleScoreBelief = (size_t)desc->scoreBeliefLen;

  for(int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
    runSingleFull(
      *desc,
      spatialInputNHWC + batchIdx * singleSpatial,
      globalInput + batchIdx * singleGlobal,
      policyPassFullOut + batchIdx * singlePolicyPass,
      policyFullOut + batchIdx * singlePolicy,
      valueOut + batchIdx * singleValue,
      miscOut + batchIdx * singleMisc,
      moreMiscOut + batchIdx * singleMoreMisc,
      ownershipOut + batchIdx * singleOwnership,
      scoringOut + batchIdx * singleScoring,
      futurePosOut + batchIdx * singleFuturePos,
      sekiOut + batchIdx * singleSeki,
      scoreBeliefOut + batchIdx * singleScoreBelief
    );
  }
}

int TransformerInferenceEngine::getModelVersion() const {
  return desc->modelVersion;
}

int TransformerInferenceEngine::getPosLen() const {
  return desc->posLen;
}

int TransformerInferenceEngine::getNumInputChannels() const {
  return desc->numInputChannels;
}

int TransformerInferenceEngine::getNumInputGlobalChannels() const {
  return desc->numInputGlobalChannels;
}

int TransformerInferenceEngine::getNumPolicyChannels() const {
  return 2;
}

int TransformerInferenceEngine::getNumFullPolicyChannels() const {
  return 6;
}

int TransformerInferenceEngine::getScoreBeliefLen() const {
  return desc->scoreBeliefLen;
}

bool TransformerInferenceEngine::supportsFullRawOutputs() const {
  return !desc->policyBoardFullWeight.empty() && !desc->policyPassFullWeight.empty();
}
