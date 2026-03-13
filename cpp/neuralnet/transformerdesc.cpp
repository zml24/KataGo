#include "../neuralnet/transformerdesc.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_map>

#include "../core/fileutils.h"
#include "../core/global.h"

using namespace std;

#if !defined(BYTE_ORDER) || (BYTE_ORDER != LITTLE_ENDIAN && BYTE_ORDER != BIG_ENDIAN)
#error Define BYTE_ORDER to be equal to either LITTLE_ENDIAN or BIG_ENDIAN
#endif

namespace {

static const char TRANSFORMER_MAGIC[] = "KGTRN001";

struct ParsedTensor {
  vector<int32_t> shape;
  vector<float> values;
};

struct Reader {
  const char* data;
  size_t size;
  size_t pos;

  Reader(const string& buf)
    : data(buf.data()), size(buf.size()), pos(0) {}

  void require(size_t n, const string& what) {
    if(pos + n > size)
      throw StringError("Transformer model parse error while reading " + what);
  }

  template<typename T>
  T readPrimitive(const string& what) {
    require(sizeof(T), what);
    T value;
    std::memcpy(&value, data + pos, sizeof(T));
    pos += sizeof(T);
#if BYTE_ORDER == BIG_ENDIAN
    char* bytes = reinterpret_cast<char*>(&value);
    std::reverse(bytes, bytes + sizeof(T));
#endif
    return value;
  }

  string readString(const string& what) {
    uint32_t len = readPrimitive<uint32_t>(what + " length");
    require(len, what);
    string s(data + pos, data + pos + len);
    pos += len;
    return s;
  }

  vector<float> readFloatVector(size_t n, const string& what) {
    require(n * sizeof(float), what);
    vector<float> out(n);
    std::memcpy(out.data(), data + pos, n * sizeof(float));
    pos += n * sizeof(float);
#if BYTE_ORDER == BIG_ENDIAN
    char* bytes = reinterpret_cast<char*>(out.data());
    for(size_t i = 0; i < n; i++) {
      std::reverse(bytes + i * sizeof(float), bytes + (i + 1) * sizeof(float));
    }
#endif
    return out;
  }
};

static void loadFileMaybeGZipped(
  const string& fileName,
  const string& expectedSha256,
  string& buf,
  string& actualSha256
) {
  if(Global::isSuffix(fileName,".gz"))
    FileUtils::uncompressAndLoadFileIntoString(fileName, expectedSha256, buf, &actualSha256);
  else
    FileUtils::loadFileIntoString(fileName, expectedSha256, buf, &actualSha256);
}

static void checkTensorShape(
  const ParsedTensor& tensor,
  const vector<int32_t>& expected,
  const string& name
) {
  if(tensor.shape != expected) {
    throw StringError("Transformer tensor " + name + " had an unexpected shape");
  }
}

static ParsedTensor takeTensor(
  unordered_map<string, ParsedTensor>& tensors,
  const string& name
) {
  auto iter = tensors.find(name);
  if(iter == tensors.end())
    throw StringError("Missing transformer tensor: " + name);
  ParsedTensor tensor = std::move(iter->second);
  tensors.erase(iter);
  return tensor;
}

}

TransformerModelDesc::TransformerModelDesc()
  : formatVersion(-1),
    modelVersion(-1),
    posLen(0),
    hiddenSize(0),
    numLayers(0),
    numHeads(0),
    headDim(0),
    ffnDim(0),
    numInputChannels(0),
    numInputGlobalChannels(0),
    stemKernelSize(0),
    scoreMode(-1),
    numScoreBeliefs(0),
    scoreBeliefLen(0),
    hasPosEmbed(false),
    postProcessParams()
{}

TransformerModelDesc::~TransformerModelDesc() {}

bool TransformerModelDesc::tryLoadFromFileMaybeGZipped(
  const string& fileName,
  const string& expectedSha256,
  TransformerModelDesc& descBuf
) {
  string buf;
  string actualSha256;
  loadFileMaybeGZipped(fileName, expectedSha256, buf, actualSha256);
  if(buf.size() < sizeof(TRANSFORMER_MAGIC) - 1)
    return false;
  if(std::memcmp(buf.data(), TRANSFORMER_MAGIC, sizeof(TRANSFORMER_MAGIC) - 1) != 0)
    return false;

  Reader reader(buf);
  reader.require(sizeof(TRANSFORMER_MAGIC) - 1, "magic");
  reader.pos += sizeof(TRANSFORMER_MAGIC) - 1;

  uint32_t formatVersion = reader.readPrimitive<uint32_t>("format version");
  if(formatVersion != 1 && formatVersion != 2)
    throw StringError("Unsupported transformer model format version: " + Global::uint64ToString(formatVersion));

  descBuf = TransformerModelDesc();
  descBuf.formatVersion = (int)formatVersion;
  descBuf.sha256 = actualSha256;
  descBuf.name = reader.readString("model name");
  descBuf.modelVersion = reader.readPrimitive<int32_t>("model version");
  descBuf.posLen = reader.readPrimitive<int32_t>("pos len");
  descBuf.hiddenSize = reader.readPrimitive<int32_t>("hidden size");
  descBuf.numLayers = reader.readPrimitive<int32_t>("num layers");
  descBuf.numHeads = reader.readPrimitive<int32_t>("num heads");
  descBuf.ffnDim = reader.readPrimitive<int32_t>("ffn dim");
  descBuf.numInputChannels = reader.readPrimitive<int32_t>("num input channels");
  descBuf.numInputGlobalChannels = reader.readPrimitive<int32_t>("num input global channels");
  descBuf.stemKernelSize = reader.readPrimitive<int32_t>("stem kernel size");
  descBuf.hasPosEmbed = reader.readPrimitive<int32_t>("has pos embed") != 0;
  if(formatVersion >= 2) {
    descBuf.scoreMode = reader.readPrimitive<int32_t>("score mode");
    descBuf.numScoreBeliefs = reader.readPrimitive<int32_t>("num score beliefs");
    descBuf.scoreBeliefLen = reader.readPrimitive<int32_t>("score belief len");
  }
  if(descBuf.posLen <= 0 || descBuf.hiddenSize <= 0 || descBuf.numLayers <= 0 || descBuf.ffnDim <= 0)
    throw StringError("Transformer model header contained non-positive dimensions");
  if(descBuf.numHeads <= 0)
    throw StringError("Transformer model header contained non-positive numHeads");
  if(descBuf.hiddenSize % descBuf.numHeads != 0)
    throw StringError("Transformer model hiddenSize must be divisible by numHeads");
  if(descBuf.stemKernelSize != 1 && descBuf.stemKernelSize != 3 && descBuf.stemKernelSize != 5)
    throw StringError("Transformer model stem kernel size must be one of 1, 3, 5");
  if(formatVersion >= 2) {
    if(descBuf.scoreMode < 0 || descBuf.scoreMode > 2)
      throw StringError("Transformer model score mode must be one of 0, 1, 2");
    if(descBuf.numScoreBeliefs <= 0)
      throw StringError("Transformer model numScoreBeliefs must be positive");
    if(descBuf.scoreBeliefLen <= 0)
      throw StringError("Transformer model scoreBeliefLen must be positive");
  }
  descBuf.headDim = descBuf.hiddenSize / descBuf.numHeads;

  descBuf.postProcessParams.tdScoreMultiplier = reader.readPrimitive<float>("td score multiplier");
  descBuf.postProcessParams.scoreMeanMultiplier = reader.readPrimitive<float>("score mean multiplier");
  descBuf.postProcessParams.scoreStdevMultiplier = reader.readPrimitive<float>("score stdev multiplier");
  descBuf.postProcessParams.leadMultiplier = reader.readPrimitive<float>("lead multiplier");
  descBuf.postProcessParams.varianceTimeMultiplier = reader.readPrimitive<float>("variance time multiplier");
  descBuf.postProcessParams.shorttermValueErrorMultiplier = reader.readPrimitive<float>("shortterm value error multiplier");
  descBuf.postProcessParams.shorttermScoreErrorMultiplier = reader.readPrimitive<float>("shortterm score error multiplier");
  descBuf.postProcessParams.outputScaleMultiplier = reader.readPrimitive<float>("output scale multiplier");

  uint32_t numTensors = reader.readPrimitive<uint32_t>("num tensors");
  unordered_map<string, ParsedTensor> tensors;
  tensors.reserve(numTensors);
  for(uint32_t i = 0; i < numTensors; i++) {
    string tensorName = reader.readString("tensor name");
    uint32_t ndim = reader.readPrimitive<uint32_t>("tensor ndim");
    vector<int32_t> shape;
    shape.reserve(ndim);
    size_t numValues = 1;
    for(uint32_t d = 0; d < ndim; d++) {
      int32_t dim = reader.readPrimitive<int32_t>("tensor dim");
      if(dim <= 0)
        throw StringError("Transformer tensor " + tensorName + " had non-positive dimension");
      shape.push_back(dim);
      numValues *= (size_t)dim;
    }
    ParsedTensor tensor;
    tensor.shape = std::move(shape);
    tensor.values = reader.readFloatVector(numValues, "tensor values");
    auto result = tensors.emplace(tensorName, std::move(tensor));
    if(!result.second)
      throw StringError("Duplicate transformer tensor name: " + tensorName);
  }

  {
    ParsedTensor tensor = takeTensor(tensors, "stem.conv.weight");
    checkTensorShape(tensor, {(int32_t)descBuf.hiddenSize, (int32_t)descBuf.numInputChannels, (int32_t)descBuf.stemKernelSize, (int32_t)descBuf.stemKernelSize}, "stem.conv.weight");
    descBuf.stemConvWeight = std::move(tensor.values);
  }
  {
    ParsedTensor tensor = takeTensor(tensors, "stem.global.weight");
    checkTensorShape(tensor, {(int32_t)descBuf.numInputGlobalChannels, (int32_t)descBuf.hiddenSize}, "stem.global.weight");
    descBuf.stemGlobalWeight = std::move(tensor.values);
  }
  if(descBuf.hasPosEmbed) {
    ParsedTensor tensor = takeTensor(tensors, "stem.pos_embed");
    checkTensorShape(tensor, {(int32_t)(descBuf.posLen * descBuf.posLen), (int32_t)descBuf.hiddenSize}, "stem.pos_embed");
    descBuf.posEmbed = std::move(tensor.values);
  }
  {
    ParsedTensor tensor = takeTensor(tensors, "rope.cos");
    checkTensorShape(tensor, {(int32_t)(descBuf.posLen * descBuf.posLen), (int32_t)descBuf.headDim}, "rope.cos");
    descBuf.ropeCos = std::move(tensor.values);
  }
  {
    ParsedTensor tensor = takeTensor(tensors, "rope.sin");
    checkTensorShape(tensor, {(int32_t)(descBuf.posLen * descBuf.posLen), (int32_t)descBuf.headDim}, "rope.sin");
    descBuf.ropeSin = std::move(tensor.values);
  }

  descBuf.blocks.resize(descBuf.numLayers);
  for(int i = 0; i < descBuf.numLayers; i++) {
    TransformerBlockDesc& block = descBuf.blocks[i];
    const string prefix = "blocks." + Global::intToString(i) + ".";

    auto readVec = [&](const string& suffix, const vector<int32_t>& shape) -> vector<float> {
      ParsedTensor tensor = takeTensor(tensors, prefix + suffix);
      checkTensorShape(tensor, shape, prefix + suffix);
      return std::move(tensor.values);
    };

    block.norm1Weight = readVec("norm1.weight", {(int32_t)descBuf.hiddenSize});
    block.qWeight = readVec("q.weight", {(int32_t)descBuf.hiddenSize, (int32_t)descBuf.hiddenSize});
    block.kWeight = readVec("k.weight", {(int32_t)descBuf.hiddenSize, (int32_t)descBuf.hiddenSize});
    block.vWeight = readVec("v.weight", {(int32_t)descBuf.hiddenSize, (int32_t)descBuf.hiddenSize});
    block.outWeight = readVec("out.weight", {(int32_t)descBuf.hiddenSize, (int32_t)descBuf.hiddenSize});
    block.norm2Weight = readVec("norm2.weight", {(int32_t)descBuf.hiddenSize});
    block.ffnW1Weight = readVec("ffn_w1.weight", {(int32_t)descBuf.hiddenSize, (int32_t)descBuf.ffnDim});
    block.ffnWGateWeight = readVec("ffn_wgate.weight", {(int32_t)descBuf.hiddenSize, (int32_t)descBuf.ffnDim});
    block.ffnW2Weight = readVec("ffn_w2.weight", {(int32_t)descBuf.ffnDim, (int32_t)descBuf.hiddenSize});
  }

  {
    ParsedTensor tensor = takeTensor(tensors, "final_norm.weight");
    checkTensorShape(tensor, {(int32_t)descBuf.hiddenSize}, "final_norm.weight");
    descBuf.finalNormWeight = std::move(tensor.values);
  }
  {
    ParsedTensor tensor = takeTensor(tensors, "policy.board.weight");
    checkTensorShape(tensor, {(int32_t)descBuf.hiddenSize, 2}, "policy.board.weight");
    descBuf.policyBoardWeight = std::move(tensor.values);
  }
  {
    ParsedTensor tensor = takeTensor(tensors, "policy.pass.weight");
    checkTensorShape(tensor, {(int32_t)descBuf.hiddenSize, 2}, "policy.pass.weight");
    descBuf.policyPassWeight = std::move(tensor.values);
  }
  {
    ParsedTensor tensor = takeTensor(tensors, "value.value.weight");
    checkTensorShape(tensor, {(int32_t)descBuf.hiddenSize, 3}, "value.value.weight");
    descBuf.valueWeight = std::move(tensor.values);
  }
  {
    ParsedTensor tensor = takeTensor(tensors, "value.score.weight");
    checkTensorShape(tensor, {(int32_t)descBuf.hiddenSize, 6}, "value.score.weight");
    descBuf.scoreValueWeight = std::move(tensor.values);
  }
  {
    ParsedTensor tensor = takeTensor(tensors, "value.ownership.weight");
    checkTensorShape(tensor, {(int32_t)descBuf.hiddenSize, 1}, "value.ownership.weight");
    descBuf.ownershipWeight = std::move(tensor.values);
  }
  if(formatVersion >= 2) {
    {
      ParsedTensor tensor = takeTensor(tensors, "policy.board_full.weight");
      checkTensorShape(tensor, {(int32_t)descBuf.hiddenSize, 6}, "policy.board_full.weight");
      descBuf.policyBoardFullWeight = std::move(tensor.values);
    }
    {
      ParsedTensor tensor = takeTensor(tensors, "policy.pass_full.weight");
      checkTensorShape(tensor, {(int32_t)descBuf.hiddenSize, 6}, "policy.pass_full.weight");
      descBuf.policyPassFullWeight = std::move(tensor.values);
    }
    {
      ParsedTensor tensor = takeTensor(tensors, "value.misc.weight");
      checkTensorShape(tensor, {(int32_t)descBuf.hiddenSize, 10}, "value.misc.weight");
      descBuf.miscWeight = std::move(tensor.values);
    }
    {
      ParsedTensor tensor = takeTensor(tensors, "value.moremisc.weight");
      checkTensorShape(tensor, {(int32_t)descBuf.hiddenSize, 8}, "value.moremisc.weight");
      descBuf.moreMiscWeight = std::move(tensor.values);
    }
    {
      ParsedTensor tensor = takeTensor(tensors, "value.scoring.weight");
      checkTensorShape(tensor, {(int32_t)descBuf.hiddenSize, 1}, "value.scoring.weight");
      descBuf.scoringWeight = std::move(tensor.values);
    }
    {
      ParsedTensor tensor = takeTensor(tensors, "value.futurepos.weight");
      checkTensorShape(tensor, {(int32_t)descBuf.hiddenSize, 2}, "value.futurepos.weight");
      descBuf.futurePosWeight = std::move(tensor.values);
    }
    {
      ParsedTensor tensor = takeTensor(tensors, "value.seki.weight");
      checkTensorShape(tensor, {(int32_t)descBuf.hiddenSize, 4}, "value.seki.weight");
      descBuf.sekiWeight = std::move(tensor.values);
    }
    if(descBuf.scoreMode == 0) {
      ParsedTensor tensor = takeTensor(tensors, "scorebelief.simple.weight");
      checkTensorShape(tensor, {(int32_t)descBuf.hiddenSize, (int32_t)descBuf.scoreBeliefLen}, "scorebelief.simple.weight");
      descBuf.scoreBeliefSimpleWeight = std::move(tensor.values);
    }
    else {
      ParsedTensor tensor = takeTensor(tensors, "scorebelief.mix.weight");
      checkTensorShape(
        tensor,
        {(int32_t)descBuf.hiddenSize, (int32_t)(descBuf.scoreBeliefLen * descBuf.numScoreBeliefs + descBuf.numScoreBeliefs)},
        "scorebelief.mix.weight"
      );
      descBuf.scoreBeliefMixWeight = std::move(tensor.values);
      if(descBuf.scoreMode == 2) {
        ParsedTensor s2off = takeTensor(tensors, "scorebelief.s2off.weight");
        checkTensorShape(s2off, {1, (int32_t)descBuf.numScoreBeliefs}, "scorebelief.s2off.weight");
        descBuf.scoreBeliefS2OffWeight = std::move(s2off.values);
        ParsedTensor s2par = takeTensor(tensors, "scorebelief.s2par.weight");
        checkTensorShape(s2par, {1, (int32_t)descBuf.numScoreBeliefs}, "scorebelief.s2par.weight");
        descBuf.scoreBeliefS2ParWeight = std::move(s2par.values);
      }
    }
  }

  if(!tensors.empty()) {
    throw StringError("Unexpected extra tensors remained while parsing transformer model");
  }
  if(reader.pos != reader.size)
    throw StringError("Unexpected trailing bytes remained while parsing transformer model");

  return true;
}
