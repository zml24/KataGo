#include "../tests/tests.h"

#include "../core/fileutils.h"
#include "../dataio/sgf.h"
#include "../external/nlohmann_json/json.hpp"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/transformerdesc.h"

//------------------------
#include "../core/using.h"
//------------------------

using json = nlohmann::json;

namespace {

struct TransformerSampleSpec {
  string name;
  string sgf;
  int turnIdx;
  bool overrideKomi;
  float komi;
};

static vector<TransformerSampleSpec> getTransformerSamples() {
  return {
    {
      "canary_opening_18",
      "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[19]KM[7]PW[White]PB[Black];B[pd];W[pp];B[dd];W[dp];B[qn];W[nq];B[cq];W[dq];B[cp];W[do];B[bn];W[cc];B[cd];W[dc];B[ec];W[eb];B[fb];W[fc];B[ed];W[gb];B[db];W[fa];B[cb];W[qo];B[pn];W[nc];B[qj];W[qc];B[qd];W[pc];B[od];W[nd];B[ne];W[me];B[mf];W[nf])",
      18,
      false,
      0.0f,
    },
    {
      "canary_opening_36",
      "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[19]KM[7]PW[White]PB[Black];B[pd];W[pp];B[dd];W[dp];B[qn];W[nq];B[cq];W[dq];B[cp];W[do];B[bn];W[cc];B[cd];W[dc];B[ec];W[eb];B[fb];W[fc];B[ed];W[gb];B[db];W[fa];B[cb];W[qo];B[pn];W[nc];B[qj];W[qc];B[qd];W[pc];B[od];W[nd];B[ne];W[me];B[mf];W[nf])",
      36,
      false,
      0.0f,
    },
    {
      "canary_fight_23",
      "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[19]KM[7]PW[White]PB[Black];B[qd];W[dd];B[pp];W[dp];B[cf];W[fc];B[nd];W[nq];B[cq];W[dq];B[cp];W[cn];B[co];W[do];B[bn];W[cm];B[bm];W[cl];B[qn];W[pq];B[qq];W[qr];B[oq])",
      23,
      false,
      0.0f,
    },
  };
}

static void fillRowForModelVersion(
  int modelVersion,
  const Board& board,
  const BoardHistory& hist,
  Player nextPla,
  const MiscNNInputParams& nnInputParams,
  int nnXLen,
  int nnYLen,
  bool inputsUseNHWC,
  float* rowSpatial,
  float* rowGlobal
) {
  if(modelVersion >= 15)
    NNInputs::fillRowV7(board,hist,nextPla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowSpatial,rowGlobal);
  else
    throw StringError("runTransformerDump 当前只支持 model version >= 15");
}

static json nnOutputToJson(const NNOutput& output, int nnXLen, int nnYLen) {
  json ret;
  ret["whiteWinProb"] = output.whiteWinProb;
  ret["whiteLossProb"] = output.whiteLossProb;
  ret["whiteNoResultProb"] = output.whiteNoResultProb;
  ret["whiteScoreMean"] = output.whiteScoreMean;
  ret["whiteScoreMeanSq"] = output.whiteScoreMeanSq;
  ret["whiteLead"] = output.whiteLead;
  ret["varTimeLeft"] = output.varTimeLeft;
  ret["shorttermWinlossError"] = output.shorttermWinlossError;
  ret["shorttermScoreError"] = output.shorttermScoreError;
  ret["policyProbs"] = vector<float>(output.policyProbs, output.policyProbs + nnXLen * nnYLen + 1);
  ret["whiteOwnerMap"] = vector<float>(output.whiteOwnerMap, output.whiteOwnerMap + nnXLen * nnYLen);
  return ret;
}

template<typename T>
static vector<T> sliceVec(const vector<T>& values, size_t offset, size_t count) {
  return vector<T>(values.begin() + offset, values.begin() + offset + count);
}

}

void Tests::runTransformerDump(
  const string& modelFile,
  const string& outputFile,
  int symmetry,
  const string& sampleFilter,
  enabled_t precisionMode,
  const string& precisionLabel
) {
  TransformerModelDesc transformerDesc;
  if(!TransformerModelDesc::tryLoadFromFileMaybeGZipped(modelFile, "", transformerDesc))
    throw StringError("runTransformerDump 只支持原生 Transformer 模型文件 (.kgtr/.kgtr.gz)");

  LoadedModel* loadedModel = NeuralNet::loadModelFile(modelFile, "");
  const ModelDesc& modelDesc = NeuralNet::getModelDesc(loadedModel);
  const int nnXLen = transformerDesc.posLen;
  const int nnYLen = transformerDesc.posLen;
  const int modelVersion = modelDesc.modelVersion;
  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  const bool inputsUseNHWC = true;

  vector<TransformerSampleSpec> sampleSpecs = getTransformerSamples();
  if(!sampleFilter.empty()) {
    vector<TransformerSampleSpec> filtered;
    for(const TransformerSampleSpec& spec : sampleSpecs) {
      if(spec.name == sampleFilter)
        filtered.push_back(spec);
    }
    if(filtered.empty())
      throw StringError("runTransformerDump sample filter did not match any sample: " + sampleFilter);
    sampleSpecs = std::move(filtered);
  }
  const int sampleCount = (int)sampleSpecs.size();

  ComputeContext* context = NeuralNet::createComputeContext(
    {},
    nullptr,
    nnXLen,
    nnYLen,
    "",
    "",
    false,
    precisionMode,
    enabled_t::True,
    loadedModel
  );
  ComputeHandle* handle = NeuralNet::createComputeHandle(
    context,
    loadedModel,
    nullptr,
    sampleCount,
    true,
    inputsUseNHWC,
    -1,
    0
  );
  InputBuffers* inputBuffers = NeuralNet::createInputBuffers(loadedModel, sampleCount, nnXLen, nnYLen);

  vector<NNResultBuf> bufStorage(sampleCount);
  vector<NNResultBuf*> inputBufPtrs(sampleCount);
  vector<NNOutput> outputStorage(sampleCount);
  vector<NNOutput*> outputPtrs(sampleCount);

  json root;
  root["modelFile"] = modelFile;
  root["modelVersion"] = modelVersion;
  root["nnXLen"] = nnXLen;
  root["nnYLen"] = nnYLen;
  root["symmetry"] = symmetry;
  root["precision"] = precisionLabel;
  root["policyOptimism"] = 0.0;
  root["postProcessParams"]["tdScoreMultiplier"] = modelDesc.postProcessParams.tdScoreMultiplier;
  root["postProcessParams"]["scoreMeanMultiplier"] = modelDesc.postProcessParams.scoreMeanMultiplier;
  root["postProcessParams"]["scoreStdevMultiplier"] = modelDesc.postProcessParams.scoreStdevMultiplier;
  root["postProcessParams"]["leadMultiplier"] = modelDesc.postProcessParams.leadMultiplier;
  root["postProcessParams"]["varianceTimeMultiplier"] = modelDesc.postProcessParams.varianceTimeMultiplier;
  root["postProcessParams"]["shorttermValueErrorMultiplier"] = modelDesc.postProcessParams.shorttermValueErrorMultiplier;
  root["postProcessParams"]["shorttermScoreErrorMultiplier"] = modelDesc.postProcessParams.shorttermScoreErrorMultiplier;
  root["samples"] = json::array();

  for(int i = 0; i < sampleCount; i++) {
    const TransformerSampleSpec& spec = sampleSpecs[i];
    std::unique_ptr<CompactSgf> sgf = CompactSgf::parse(spec.sgf);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = sgf->getRulesOrFail();
    sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, spec.turnIdx);
    if(spec.overrideKomi)
      hist.setKomi(spec.komi);

    MiscNNInputParams nnInputParams;
    nnInputParams.symmetry = symmetry;

    NNResultBuf& buf = bufStorage[i];
    buf.boardXSizeForServer = board.x_size;
    buf.boardYSizeForServer = board.y_size;
    buf.symmetry = symmetry;
    buf.policyOptimism = 0.0;
    buf.hasRowMeta = false;
    buf.rowSpatialBuf.resize((size_t)numSpatialFeatures * nnXLen * nnYLen);
    buf.rowGlobalBuf.resize(numGlobalFeatures);
    buf.rowMetaBuf.clear();
    fillRowForModelVersion(
      modelVersion,
      board,
      hist,
      nextPla,
      nnInputParams,
      nnXLen,
      nnYLen,
      inputsUseNHWC,
      buf.rowSpatialBuf.data(),
      buf.rowGlobalBuf.data()
    );
    inputBufPtrs[i] = &buf;

    NNOutput& output = outputStorage[i];
    output.nnXLen = nnXLen;
    output.nnYLen = nnYLen;
    output.whiteOwnerMap = new float[nnXLen * nnYLen];
    output.noisedPolicyProbs = nullptr;
    outputPtrs[i] = &output;

    json sampleJson;
    sampleJson["name"] = spec.name;
    sampleJson["sgf"] = spec.sgf;
    sampleJson["turnIdx"] = spec.turnIdx;
    sampleJson["nextPla"] = PlayerIO::playerToString(nextPla);
    sampleJson["spatialNHWC"] = buf.rowSpatialBuf;
    sampleJson["global"] = buf.rowGlobalBuf;
    root["samples"].push_back(std::move(sampleJson));
  }

  TransformerRawOutputs rawOutputs;
  if(!NeuralNet::getTransformerRawOutputs(handle, inputBuffers, sampleCount, inputBufPtrs.data(), rawOutputs))
    throw StringError("当前 backend 没有提供 Transformer raw output dump 支持");

  root["numPolicyChannels"] = rawOutputs.numPolicyChannels;
  root["numFullPolicyChannels"] = rawOutputs.numFullPolicyChannels;
  root["numMiscChannels"] = rawOutputs.numMiscChannels;
  root["numMoreMiscChannels"] = rawOutputs.numMoreMiscChannels;
  root["numScoringChannels"] = rawOutputs.numScoringChannels;
  root["numFuturePosChannels"] = rawOutputs.numFuturePosChannels;
  root["numSekiChannels"] = rawOutputs.numSekiChannels;
  root["scoreBeliefLen"] = rawOutputs.scoreBeliefLen;

  NeuralNet::getOutput(handle, inputBuffers, sampleCount, inputBufPtrs.data(), outputPtrs);

  for(int i = 0; i < sampleCount; i++) {
    json& sampleJson = root["samples"][i];
    const size_t seqLen = (size_t)nnXLen * nnYLen;
    sampleJson["raw"]["policyPass"] = sliceVec(rawOutputs.policyPass, (size_t)i * rawOutputs.numPolicyChannels, rawOutputs.numPolicyChannels);
    sampleJson["raw"]["policy"] = sliceVec(rawOutputs.policy, (size_t)i * seqLen * rawOutputs.numPolicyChannels, seqLen * rawOutputs.numPolicyChannels);
    sampleJson["raw"]["value"] = sliceVec(rawOutputs.value, (size_t)i * 3, 3);
    sampleJson["raw"]["scoreValue"] = sliceVec(rawOutputs.scoreValue, (size_t)i * 6, 6);
    sampleJson["raw"]["ownership"] = sliceVec(rawOutputs.ownership, (size_t)i * seqLen, seqLen);
    if(rawOutputs.numFullPolicyChannels > 0) {
      sampleJson["rawFull"]["policyPass"] = sliceVec(rawOutputs.fullPolicyPass, (size_t)i * rawOutputs.numFullPolicyChannels, rawOutputs.numFullPolicyChannels);
      sampleJson["rawFull"]["policy"] = sliceVec(rawOutputs.fullPolicy, (size_t)i * seqLen * rawOutputs.numFullPolicyChannels, seqLen * rawOutputs.numFullPolicyChannels);
      sampleJson["rawFull"]["value"] = sliceVec(rawOutputs.value, (size_t)i * 3, 3);
      sampleJson["rawFull"]["misc"] = sliceVec(rawOutputs.misc, (size_t)i * rawOutputs.numMiscChannels, rawOutputs.numMiscChannels);
      sampleJson["rawFull"]["moreMisc"] = sliceVec(rawOutputs.moreMisc, (size_t)i * rawOutputs.numMoreMiscChannels, rawOutputs.numMoreMiscChannels);
      sampleJson["rawFull"]["ownership"] = sliceVec(rawOutputs.ownership, (size_t)i * seqLen, seqLen);
      sampleJson["rawFull"]["scoring"] = sliceVec(rawOutputs.scoring, (size_t)i * seqLen * rawOutputs.numScoringChannels, seqLen * rawOutputs.numScoringChannels);
      sampleJson["rawFull"]["futurePos"] = sliceVec(rawOutputs.futurePos, (size_t)i * seqLen * rawOutputs.numFuturePosChannels, seqLen * rawOutputs.numFuturePosChannels);
      sampleJson["rawFull"]["seki"] = sliceVec(rawOutputs.seki, (size_t)i * seqLen * rawOutputs.numSekiChannels, seqLen * rawOutputs.numSekiChannels);
      sampleJson["rawFull"]["scoreBelief"] = sliceVec(rawOutputs.scoreBelief, (size_t)i * rawOutputs.scoreBeliefLen, rawOutputs.scoreBeliefLen);
    }
    sampleJson["nnOutput"] = nnOutputToJson(outputStorage[i], nnXLen, nnYLen);
  }

  std::ofstream out;
  FileUtils::open(out, outputFile);
  out << root.dump(2);
  out.close();

  for(int i = 0; i < sampleCount; i++) {
    delete[] outputStorage[i].whiteOwnerMap;
    outputStorage[i].whiteOwnerMap = nullptr;
  }

  NeuralNet::freeInputBuffers(inputBuffers);
  NeuralNet::freeComputeHandle(handle);
  NeuralNet::freeComputeContext(context);
  NeuralNet::freeLoadedModel(loadedModel);
}
