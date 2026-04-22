#include "../neuralnet/nneval.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/serve_profiler.h"

#include <chrono>

using namespace std;

//-------------------------------------------------------------------------------------

NNResultBuf::NNResultBuf()
  : clientWaitingForResult(),
    resultMutex(),
    hasResult(false),
    includeOwnerMap(false),
    boardXSizeForServer(0),
    boardYSizeForServer(0),
    rowSpatialBuf(),
    rowGlobalBuf(),
    rowMetaBuf(),
    hasRowMeta(false),
    result(nullptr),
    errorLogLockout(false),
    // If no symmetry is specified, it will use default or random based on config.
    symmetry(NNInputs::SYMMETRY_NOTSPECIFIED),
    policyOptimism(0.0)
{}

NNResultBuf::~NNResultBuf() {
}

//-------------------------------------------------------------------------------------

NNServerBuf::NNServerBuf(const NNEvaluator& nnEval, const LoadedModel* model)
  :inputBuffers(NULL)
{
  int maxBatchSize = nnEval.getMaxBatchSize();
  if(model != NULL)
    inputBuffers = NeuralNet::createInputBuffers(model,maxBatchSize,nnEval.getNNXLen(),nnEval.getNNYLen());
}

NNServerBuf::~NNServerBuf() {
  if(inputBuffers != NULL)
    NeuralNet::freeInputBuffers(inputBuffers);
  inputBuffers = NULL;
}

//-------------------------------------------------------------------------------------

NNEvaluator::NNEvaluator(
  const string& mName,
  const string& mFileName,
  const string& expectedSha256,
  Logger* lg,
  int maxBatchSz,
  int xLen,
  int yLen,
  bool rExactNNLen,
  bool iUseNHWC,
  int nnCacheSizePowerOfTwo,
  int nnMutexPoolSizePowerofTwo,
  bool skipNeuralNet,
  const string& openCLTunerFile,
  const string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  int numThr,
  const vector<int>& gpuIdxByServerThr,
  const string& rSeed,
  bool doRandomize,
  int defaultSymmetry
)
  :modelName(mName),
   modelFileName(mFileName),
   nnXLen(xLen),
   nnYLen(yLen),
   requireExactNNLen(rExactNNLen),
   policySize(NNPos::getPolicySize(xLen,yLen)),
   inputsUseNHWC(iUseNHWC),
   usingFP16Mode(useFP16Mode),
   usingNHWCMode(useNHWCMode),
   numThreads(numThr),
   gpuIdxByServerThread(gpuIdxByServerThr),
   randSeed(rSeed),
   debugSkipNeuralNet(skipNeuralNet),
   computeContext(NULL),
   loadedModel(NULL),
   nnCacheTable(NULL),
   logger(lg),
   internalModelName(),
   modelVersion(-1),
   inputsVersion(-1),
   numInputMetaChannels(0),
   postProcessParams(),
   numServerThreadsEverSpawned(0),
   serverThreads(),
   maxBatchSize(maxBatchSz),
   m_numRowsProcessed(0),
   m_numBatchesProcessed(0),
   bufferMutex(),
   isKilled(false),
   numServerThreadsStartingUp(0),
   mainThreadWaitingForSpawn(),
   numOngoingEvals(0),
   numWaitingEvals(0),
   numEvalsToAwaken(0),
   drainPending(false),
   waitingForFinish(),
   currentDoRandomize(doRandomize),
   currentDefaultSymmetry(defaultSymmetry),
   currentBatchSize(maxBatchSz),
   isSuperseded(false),
   nnOutputPool(std::make_shared<NNOutputPool>()),
   queryQueue()
{
  if(nnXLen > NNPos::MAX_BOARD_LEN)
    throw StringError("Maximum supported nnEval board size is " + Global::intToString(NNPos::MAX_BOARD_LEN));
  if(nnYLen > NNPos::MAX_BOARD_LEN)
    throw StringError("Maximum supported nnEval board size is " + Global::intToString(NNPos::MAX_BOARD_LEN));
  if(maxBatchSize <= 0)
    throw StringError("maxBatchSize is negative: " + Global::intToString(maxBatchSize));
  if(gpuIdxByServerThread.size() != numThreads)
    throw StringError("gpuIdxByServerThread.size() != numThreads");

  if(logger != NULL) {
    logger->write(
      "Initializing neural net buffer to be size " +
      Global::intToString(nnXLen) + " * " + Global::intToString(nnYLen) +
      (requireExactNNLen ? " exactly" : " allowing smaller boards")
    );
  }

  if(nnCacheSizePowerOfTwo >= 0)
    nnCacheTable = new NNCacheTable(nnCacheSizePowerOfTwo, nnMutexPoolSizePowerofTwo);

  if(!debugSkipNeuralNet) {
    vector<int> gpuIdxs = gpuIdxByServerThread;
    std::sort(gpuIdxs.begin(), gpuIdxs.end());
    auto last = std::unique(gpuIdxs.begin(), gpuIdxs.end());
    gpuIdxs.erase(last,gpuIdxs.end());
    loadedModel = NeuralNet::loadModelFile(modelFileName,expectedSha256);
    const ModelDesc& desc = NeuralNet::getModelDesc(loadedModel);
    internalModelName = desc.name;
    modelVersion = desc.modelVersion;
    inputsVersion = NNModelVersion::getInputsVersion(modelVersion);
    numInputMetaChannels = desc.numInputMetaChannels;
    postProcessParams = desc.postProcessParams;
    computeContext = NeuralNet::createComputeContext(
      gpuIdxs,logger,nnXLen,nnYLen,
      openCLTunerFile,homeDataDirOverride,openCLReTunePerBoardSize,
      usingFP16Mode,usingNHWCMode,loadedModel
    );
  }
  else {
    internalModelName = "random";
    modelVersion = NNModelVersion::defaultModelVersion;
    inputsVersion = NNModelVersion::getInputsVersion(modelVersion);
  }

  //Reserve a decent amount above the batch size so that allocation is unlikely.
  queryQueue.reserve(maxBatchSize * 4 * gpuIdxByServerThread.size());
  //Starts readonly. Becomes writable once we spawn server threads
  queryQueue.setReadOnly();
}

static void fillDummyResult(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  int nnXLen,
  int nnYLen,
  int policySize,
  double policyOptimism,
  bool includeOwnerMap,
  NNResultBuf& buf
) {
  std::lock_guard<std::mutex> resultLock(buf.resultMutex);
  buf.result = std::make_shared<NNOutput>();
  buf.result->nnXLen = nnXLen;
  buf.result->nnYLen = nnYLen;
  buf.result->whiteWinProb = 0.5f;
  buf.result->whiteLossProb = 0.5f;
  buf.result->whiteNoResultProb = 0.0f;
  buf.result->whiteScoreMean = 0.0f;
  buf.result->whiteScoreMeanSq = 0.0f;
  buf.result->whiteLead = 0.0f;
  buf.result->varTimeLeft = (float)(0.5 * board.x_size * board.y_size);
  buf.result->shorttermWinlossError = 0.0f;
  buf.result->shorttermScoreError = 0.0f;
  buf.result->policyOptimismUsed = (float)policyOptimism;

  if(includeOwnerMap) {
    float* whiteOwnerMap = new float[nnXLen * nnYLen];
    std::fill(whiteOwnerMap, whiteOwnerMap + nnXLen * nnYLen, 0.0f);
    buf.result->whiteOwnerMap = whiteOwnerMap;
  }
  else {
    buf.result->whiteOwnerMap = NULL;
  }

  std::fill(buf.result->policyProbs, buf.result->policyProbs + NNPos::MAX_NN_POLICY_SIZE, 0.0f);
  int legalCount = 0;
  for(int pos = 0; pos < policySize; pos++) {
    Loc loc = NNPos::posToLoc(pos, board.x_size, board.y_size, nnXLen, nnYLen);
    if(history.isLegal(board, loc, nextPlayer))
      legalCount += 1;
  }

  if(legalCount > 0) {
    float uniformProb = 1.0f / legalCount;
    for(int pos = 0; pos < policySize; pos++) {
      Loc loc = NNPos::posToLoc(pos, board.x_size, board.y_size, nnXLen, nnYLen);
      buf.result->policyProbs[pos] = history.isLegal(board, loc, nextPlayer) ? uniformProb : 0.0f;
    }
  }
  else {
    int passPos = NNPos::locToPos(Board::PASS_LOC, board.x_size, nnXLen, nnYLen);
    buf.result->policyProbs[passPos] = 1.0f;
  }

  buf.hasResult = true;
}

void NNEvaluator::finishOngoingEval() {
  std::lock_guard<std::mutex> lock(bufferMutex);
  numOngoingEvals -= 1;
  //Only broadcast when somebody is actually parked on waitingForFinish.
  //Without this gate, every evaluate() completion wakes all 400 game threads
  //(thundering herd) and costs ~2.4x throughput in RL selfplay.
  //
  //The only live consumer now is killServerThreads()'s post-join drain,
  //which sets drainPending=true while spinning on numOngoingEvals==0.
  //waitForNextNNEvalIfAny() used to register itself here via
  //++numWaitingEvals, but is now a plain sleep_for (no condvar), so
  //numWaitingEvals stays at 0. The numWaitingEvals check is kept as a
  //defensive guard in case a future change reintroduces condvar-based
  //throttling.
  if(numWaitingEvals > 0 || drainPending)
    waitingForFinish.notify_all();
}

void NNEvaluator::releaseHeavyResources() {
  isSuperseded.store(true, std::memory_order_release);
}

void NNEvaluator::releaseRetiredBackendResources() {
  killServerThreads();
  if(nnCacheTable != NULL) {
    delete nnCacheTable;
    nnCacheTable = NULL;
  }
  releaseRetiredModelResources();
}

void NNEvaluator::releaseRetiredModelResources() {
  if(computeContext != NULL) {
    NeuralNet::freeComputeContext(computeContext);
    computeContext = NULL;
  }
  if(loadedModel != NULL) {
    NeuralNet::freeLoadedModel(loadedModel);
    loadedModel = NULL;
  }
}

bool NNEvaluator::isReleasedOrSuperseded() const {
  return isSuperseded.load(std::memory_order_acquire);
}

NNEvaluator::~NNEvaluator() {
  killServerThreads();
  releaseRetiredModelResources();

  if(nnCacheTable != NULL)
    delete nnCacheTable;
  nnCacheTable = NULL;
}

//These three names are mutated by reloadLoadedModel() under bufferMutex, so
//concurrent readers (e.g. selfplay log lines) must take the same lock to avoid
//a std::string data race during in-place model swaps.
string NNEvaluator::getModelName() const {
  std::lock_guard<std::mutex> lock(bufferMutex);
  return modelName;
}
string NNEvaluator::getModelFileName() const {
  std::lock_guard<std::mutex> lock(bufferMutex);
  return modelFileName;
}
string NNEvaluator::getInternalModelName() const {
  std::lock_guard<std::mutex> lock(bufferMutex);
  return internalModelName;
}

static bool tryAbbreviateStepString(const string& input, string& buf) {
  size_t i = 0;
  while(i < input.length() && !Global::isDigit(input[i]))
    i++;
  if(i > 1)
    return false;

  string prefix = input.substr(0, i);
  int64_t number;
  bool suc = Global::tryStringToInt64(input.substr(i),number);
  if(!suc)
    return false;

  if(number >= 10000000000LL)
    buf = prefix + std::to_string(number / 1000000000LL) + "G";
  if(number >= 10000000)
    buf = prefix + std::to_string(number / 1000000) + "M";
  else if(number >= 10000)
    buf = prefix + std::to_string(number / 1000) + "K";
  else
    buf = input;
  return true;
}

string NNEvaluator::getAbbrevInternalModelName() const {
  string name = getInternalModelName();
  std::vector<string> pieces = Global::split(name,'-');
  std::vector<string> newPieces;
  for(const string& piece: pieces) {
    string buf;
    if(piece == "kata1") {
      // skip
    }
    else if(piece.size() > 1 && piece[0] == 's' && tryAbbreviateStepString(piece,buf)) {
      newPieces.push_back(buf);
    }
    else if(piece.size() > 1 && piece[0] == 'd' && tryAbbreviateStepString(piece,buf)) {
      // skip
    }
    else {
      newPieces.push_back(piece);
    }
  }
  return Global::concat(newPieces,"-");
}

Logger* NNEvaluator::getLogger() {
  return logger;
}
bool NNEvaluator::isNeuralNetLess() const {
  return debugSkipNeuralNet;
}
int NNEvaluator::getMaxBatchSize() const {
  return maxBatchSize;
}
int NNEvaluator::getCurrentBatchSize() const {
  return currentBatchSize.load(std::memory_order_acquire);
}
void NNEvaluator::setCurrentBatchSize(int batchSize) {
  if(batchSize <= 0 || batchSize > maxBatchSize)
    throw StringError("Invalid setting for batch size");
  currentBatchSize.store(batchSize,std::memory_order_release);
}
bool NNEvaluator::requiresSGFMetadata() const {
  //numInputMetaChannels is overwritten by reloadLoadedModel under bufferMutex,
  //so observe it under the same lock to avoid a data race.
  std::lock_guard<std::mutex> lk(bufferMutex);
  return numInputMetaChannels > 0;
}

int NNEvaluator::getNumGpus() const {
#ifdef USE_EIGEN_BACKEND
  return 1;
#else
  std::set<int> gpuIdxs;
  for(int i = 0; i<gpuIdxByServerThread.size(); i++) {
    gpuIdxs.insert(gpuIdxByServerThread[i]);
  }
  return (int)gpuIdxs.size();
#endif
}
int NNEvaluator::getNumServerThreads() const {
  return (int)gpuIdxByServerThread.size();
}
std::set<int> NNEvaluator::getGpuIdxs() const {
  std::set<int> gpuIdxs;
#ifdef USE_EIGEN_BACKEND
  gpuIdxs.insert(0);
#else
  for(int i = 0; i<gpuIdxByServerThread.size(); i++) {
    gpuIdxs.insert(gpuIdxByServerThread[i]);
  }
#endif
  return gpuIdxs;
}

int NNEvaluator::getNNXLen() const {
  return nnXLen;
}
int NNEvaluator::getNNYLen() const {
  return nnYLen;
}
int NNEvaluator::getModelVersion() const {
  //Replaced by reloadLoadedModel under bufferMutex.
  std::lock_guard<std::mutex> lk(bufferMutex);
  return modelVersion;
}
double NNEvaluator::getTrunkSpatialConvDepth() const {
  //loadedModel is swapped (old freed, new installed) by reloadLoadedModel
  //under bufferMutex; read + use it under the same lock so the underlying
  //LoadedModel cannot be freed while we are inspecting its descriptor.
  std::lock_guard<std::mutex> lk(bufferMutex);
  if(loadedModel == NULL)
    return 0.0;
  return NeuralNet::getModelDesc(loadedModel).getTrunkSpatialConvDepth();
}

enabled_t NNEvaluator::getUsingFP16Mode() const {
  return usingFP16Mode;
}
enabled_t NNEvaluator::getUsingNHWCMode() const {
  return usingNHWCMode;
}

bool NNEvaluator::supportsShorttermError() const {
  //modelVersion is replaced by reloadLoadedModel under bufferMutex.
  //search.cpp's MCTS terminal-leaf hot path calls this, so the lock must
  //stay narrow; the read is a single int load.
  std::lock_guard<std::mutex> lk(bufferMutex);
  return modelVersion >= 9;
}

bool NNEvaluator::getDoRandomize() const {
  return currentDoRandomize.load(std::memory_order_acquire);
}
int NNEvaluator::getDefaultSymmetry() const {
  return currentDefaultSymmetry.load(std::memory_order_acquire);
}
void NNEvaluator::setDoRandomize(bool b) {
  currentDoRandomize.store(b, std::memory_order_release);
}
void NNEvaluator::setDefaultSymmetry(int s) {
  currentDefaultSymmetry.store(s, std::memory_order_release);
}

Rules NNEvaluator::getSupportedRules(const Rules& desiredRules, bool& supported) {
  //Same reasoning as getTrunkSpatialConvDepth: loadedModel is swapped +
  //old-freed by reloadLoadedModel under bufferMutex.
  std::lock_guard<std::mutex> lk(bufferMutex);
  if(loadedModel == NULL) {
    supported = true;
    return desiredRules;
  }
  return NeuralNet::getModelDesc(loadedModel).getSupportedRules(desiredRules, supported);
}

uint64_t NNEvaluator::numRowsProcessed() const {
  return m_numRowsProcessed.load(std::memory_order_relaxed);
}
uint64_t NNEvaluator::numBatchesProcessed() const {
  return m_numBatchesProcessed.load(std::memory_order_relaxed);
}
double NNEvaluator::averageProcessedBatchSize() const {
  return (double)numRowsProcessed() / (double)numBatchesProcessed();
}

void NNEvaluator::clearStats() {
  m_numRowsProcessed.store(0);
  m_numBatchesProcessed.store(0);
}

void NNEvaluator::clearCache() {
  if(nnCacheTable != NULL)
    nnCacheTable->clear();
}


bool NNEvaluator::isAnyThreadUsingFP16() const {
  lock_guard<std::mutex> lock(bufferMutex);
  for(const int& isUsingFP16: serverThreadsIsUsingFP16) {
    if(isUsingFP16)
      return true;
  }
  return false;
}

static void serveEvals(
  string randSeedThisThread,
  NNEvaluator* nnEval, const LoadedModel* loadedModel,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  NNServerBuf* buf = new NNServerBuf(*nnEval,loadedModel);
  Rand rand(randSeedThisThread);

  //Used to have a try catch around this but actually we're in big trouble if this raises an exception
  //and causes possibly the only nnEval thread to die, so actually go ahead and let the exception escape to
  //toplevel for easier debugging
  nnEval->serve(*buf,rand,gpuIdxForThisThread,serverThreadIdx);
  delete buf;
}

void NNEvaluator::setNumThreads(const vector<int>& gpuIdxByServerThr) {
  if(serverThreads.size() != 0)
    throw StringError("NNEvaluator::setNumThreads called when threads were already running!");
  numThreads = (int)gpuIdxByServerThr.size();
  gpuIdxByServerThread = gpuIdxByServerThr;
}

void NNEvaluator::spawnServerThreads() {
  if(serverThreads.size() != 0)
    throw StringError("NNEvaluator::spawnServerThreads called when threads were already running!");

  {
    lock_guard<std::mutex> lock(bufferMutex);
    serverThreadsIsUsingFP16.resize(numThreads,0);
  }

  queryQueue.unsetReadOnly();

  numServerThreadsStartingUp = numThreads;
  for(int i = 0; i<numThreads; i++) {
    int gpuIdxForThisThread = gpuIdxByServerThread[i];
    string randSeedThisThread = randSeed + ":NNEvalServerThread:" + Global::intToString(numServerThreadsEverSpawned);
    numServerThreadsEverSpawned++;
    std::thread* thread = new std::thread(
      &serveEvals,randSeedThisThread,this,loadedModel,gpuIdxForThisThread,i
    );
    serverThreads.push_back(thread);
  }

  unique_lock<std::mutex> lock(bufferMutex);
  while(numServerThreadsStartingUp > 0)
    mainThreadWaitingForSpawn.wait(lock);
}

void NNEvaluator::killServerThreads() {
  unique_lock<std::mutex> lock(bufferMutex);
  isKilled = true;
  lock.unlock();
  queryQueue.setReadOnly();

  waitingForFinish.notify_all();

  for(size_t i = 0; i<serverThreads.size(); i++)
    serverThreads[i]->join();
  for(size_t i = 0; i<serverThreads.size(); i++)
    delete serverThreads[i];
  serverThreads.clear();
  serverThreadsIsUsingFP16.clear();

  //Drain any evaluate() calls that got past the isSuperseded gate before we
  //set the query queue readonly. Their forcePush() failed and the fallback
  //path (see evaluate()) decrements numOngoingEvals and notifies here. Without
  //this wait, the asserts below race with the fallback and fire in Debug builds.
  //drainPending tells finishOngoingEval()'s gated notify_all to broadcast on
  //waitingForFinish even when no waitForNextNNEvalIfAny() waiter is
  //registered; otherwise the gate would skip the wakeup and we'd hang here.
  lock.lock();
  drainPending = true;
  while(numOngoingEvals > 0)
    waitingForFinish.wait(lock);
  drainPending = false;
  lock.unlock();

  //Can unset now that threads are dead and no evaluate() is mid-push
  isKilled = false;

  assert(numOngoingEvals == 0);
  assert(numWaitingEvals == 0);
  assert(numEvalsToAwaken == 0);
}

bool NNEvaluator::reloadLoadedModel(
  const string& newModelFileName,
  const string& newModelName,
  const string& expectedSha256
) {
  if(debugSkipNeuralNet)
    return false;

  // 1) Try loading the new model file BEFORE touching anything live. If the model
  //    file is corrupt (NaN/inf weights, parse error, etc.), loadModelFile throws
  //    and we return false with the existing evaluator completely untouched. Callers
  //    fall back to creating a new NNEvaluator, or just stay on the current one.
  //    This costs a few hundred MB of transient memory (two LoadedModels alive briefly)
  //    but avoids the cascade where a bad new-model crashes the whole selfplay process.
  LoadedModel* newLoadedModel = nullptr;
  try {
    newLoadedModel = NeuralNet::loadModelFile(newModelFileName, expectedSha256);
  } catch(const std::exception& e) {
    if(logger != NULL)
      logger->write(string("In-place reload rejected new model ") + newModelName + ": " + e.what());
    return false;
  }
  if(newLoadedModel == nullptr) {
    if(logger != NULL)
      logger->write("In-place reload rejected new model " + newModelName + ": loadModelFile returned null");
    return false;
  }

  // 2) Flip isSuperseded BEFORE killServerThreads so any evaluate() calls arriving
  //    during the reload fall through to fillDummyResult and never push into the
  //    queue or hit assert(!isKilled). This is the only way to safely kill/respawn
  //    server threads while game threads are potentially still calling evaluate().
  isSuperseded.store(true, std::memory_order_release);

  // Historically this woke condvar waiters parked in waitForNextNNEvalIfAny()
  // so they could re-check and exit once killServerThreads() below set
  // isKilled. Now that waitForNextNNEvalIfAny() is a plain sleep_for,
  // numWaitingEvals stays at 0 and the block is a no-op at runtime. Kept
  // as a defensive edge in case a future change reintroduces condvar-based
  // throttling.
  {
    lock_guard<std::mutex> lk(bufferMutex);
    if(numWaitingEvals > 0)
      waitingForFinish.notify_all();
  }

  // 3) Drain the server threads (they will process any already-pushed requests and
  //    exit; killServerThreads asserts numOngoingEvals==0 on return, which is
  //    guaranteed because new evaluate() calls now short-circuit into dummy results).
  killServerThreads();

  // 4+5) Swap loadedModel AND replace all snapshotted fields under a single
  //      bufferMutex critical section. This serializes with:
  //        * evaluate()'s snapshot block (same lock), so a single evaluate()
  //          always sees a consistent {loadedModel, modelVersion,
  //          inputsVersion, numInputMetaChannels, postProcessParams,
  //          modelName, modelFileName} tuple belonging to one model.
  //        * all public getters that observe loadedModel / modelVersion /
  //          numInputMetaChannels (getModelVersion, getTrunkSpatialConvDepth,
  //          getSupportedRules, requiresSGFMetadata, supportsShorttermError),
  //          which also take bufferMutex, so the old LoadedModel cannot be
  //          freed out from under a getter inspecting its descriptor.
  //
  //      killServerThreads() returned with numOngoingEvals==0, so no
  //      evaluate() is mid-snapshot; holding the lock through
  //      freeLoadedModel keeps getters blocked but cannot deadlock against
  //      server threads (none are alive here).
  {
    lock_guard<std::mutex> lk(bufferMutex);
    if(loadedModel != NULL) {
      NeuralNet::freeLoadedModel(loadedModel);
      loadedModel = NULL;
    }
    loadedModel = newLoadedModel;
    const ModelDesc& desc = NeuralNet::getModelDesc(loadedModel);
    internalModelName = desc.name;
    modelVersion = desc.modelVersion;
    inputsVersion = NNModelVersion::getInputsVersion(modelVersion);
    numInputMetaChannels = desc.numInputMetaChannels;
    postProcessParams = desc.postProcessParams;
    modelName = newModelName;
    modelFileName = newModelFileName;
  }

  // 6) Old cached NN outputs belong to old weights; drop them.
  if(nnCacheTable != NULL)
    nnCacheTable->clear();

  // 7) Respawn server threads; each one will create a fresh ComputeHandle off the
  //    new loadedModel (TRT refit cache will make this ~1s instead of minutes).
  spawnServerThreads();

  // 8) Resume normal inference.
  isSuperseded.store(false, std::memory_order_release);
  if(logger != NULL)
    logger->write("NNEvaluator in-place reloaded to " + modelName);
  return true;
}

void NNEvaluator::serve(
  NNServerBuf& buf, Rand& rand,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  int64_t numBatchesHandledThisThread = 0;
  int64_t numRowsHandledThisThread = 0;

  ComputeHandle* gpuHandle = NULL;
  if(loadedModel != NULL) {
    auto tCreateHandle = std::chrono::steady_clock::now();
    gpuHandle = NeuralNet::createComputeHandle(
      computeContext,
      loadedModel,
      logger,
      maxBatchSize,
      requireExactNNLen,
      inputsUseNHWC,
      gpuIdxForThisThread,
      serverThreadIdx
    );
    double chMs = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - tCreateHandle).count();
    if(logger) logger->write("TIMING: createComputeHandle = " + Global::doubleToString(chMs) + " ms on GPU " + Global::intToString(gpuIdxForThisThread));
  }

  {
    lock_guard<std::mutex> lock(bufferMutex);
    assert(serverThreadIdx < serverThreadsIsUsingFP16.size());
    serverThreadsIsUsingFP16[serverThreadIdx] = gpuHandle == NULL ? 0 : NeuralNet::isUsingFP16(gpuHandle) ? 1 : 0;
    numServerThreadsStartingUp--;
    if(numServerThreadsStartingUp <= 0)
      mainThreadWaitingForSpawn.notify_all();
  }

  vector<NNResultBuf*> resultBufs;
  resultBufs.reserve(maxBatchSize);

  vector<NNOutput*> outputBuf;

  // Overlap notify: previous batch's results notified during next batch's GPU computation
  vector<NNResultBuf*> prevOverlapResultBufs;
  vector<NNOutput*> prevOverlapOutputBuf;
  int prevOverlapNumRows = 0;

  unique_lock<std::mutex> lock(bufferMutex,std::defer_lock);
  initServeProfiler();
  const bool _profEnabled = g_serveProfileEnabled.load(std::memory_order_relaxed);
  ServeProfileStats _prof;
  while(true) {
    ServeProfileStats::TP _tCycleStart{}, _tAfterPop{}, _tAfterAlloc{};
    ServeProfileStats::TP _tBeforeGO{}, _tAfterGO{}, _tBeforeNotify{}, _tAfterNotify{};
    if(_profEnabled) _tCycleStart = ServeProfileStats::Clock::now();
    resultBufs.clear();
    int desiredBatchSize = std::min(maxBatchSize, currentBatchSize.load(std::memory_order_acquire));
    bool gotAnything = queryQueue.waitPopUpToN(resultBufs,desiredBatchSize);
    //Queue being closed is a signal that we're done.
    if(!gotAnything) {
      // Flush last batch's deferred notify before exiting
      if(prevOverlapNumRows > 0) {
        std::shared_ptr<NNOutputPool> pool = nnOutputPool;
        for(int row = 0; row < prevOverlapNumRows; row++) {
          NNResultBuf* rb = prevOverlapResultBufs[row];
          auto rp = std::shared_ptr<NNOutput>(prevOverlapOutputBuf[row], [pool](NNOutput* p) { pool->release(p); });
          { lock_guard<std::mutex> rl(rb->resultMutex); rb->result = std::move(rp); rb->hasResult = true; }
          rb->clientWaitingForResult.notify_one();
        }
        prevOverlapNumRows = 0;
      }
      break;
    }

    if(_profEnabled) _tAfterPop = ServeProfileStats::Clock::now();
    int numRows = (int)resultBufs.size();
    assert(numRows > 0);

    bool doRandomize = currentDoRandomize.load(std::memory_order_acquire);
    int defaultSymmetry = currentDefaultSymmetry.load(std::memory_order_acquire);

    if(debugSkipNeuralNet) {
      for(int row = 0; row < numRows; row++) {
        assert(resultBufs[row] != NULL);
        NNResultBuf* resultBuf = resultBufs[row];
        resultBufs[row] = NULL;

        int boardXSize = resultBuf->boardXSizeForServer;
        int boardYSize = resultBuf->boardYSizeForServer;

        unique_lock<std::mutex> resultLock(resultBuf->resultMutex);
        assert(resultBuf->hasResult == false);
        resultBuf->result = std::make_shared<NNOutput>();

        float* policyProbs = resultBuf->result->policyProbs;
        for(int i = 0; i<NNPos::MAX_NN_POLICY_SIZE; i++)
          policyProbs[i] = 0;

        //At this point, these aren't probabilities, since this is before the postprocessing
        //that happens for each result. These just need to be unnormalized log probabilities.
        //Illegal move filtering happens later.
        for(int y = 0; y<boardYSize; y++) {
          for(int x = 0; x<boardXSize; x++) {
            int pos = NNPos::xyToPos(x,y,nnXLen);
            policyProbs[pos] = (float)rand.nextGaussian();
          }
        }
        policyProbs[NNPos::locToPos(Board::PASS_LOC,boardXSize,nnXLen,nnYLen)] = (float)rand.nextGaussian();

        resultBuf->result->nnXLen = nnXLen;
        resultBuf->result->nnYLen = nnYLen;
        if(resultBuf->includeOwnerMap) {
          float* whiteOwnerMap = new float[nnXLen*nnYLen];
          for(int i = 0; i<nnXLen*nnYLen; i++)
            whiteOwnerMap[i] = 0.0;
          for(int y = 0; y<boardYSize; y++) {
            for(int x = 0; x<boardXSize; x++) {
              int pos = NNPos::xyToPos(x,y,nnXLen);
              whiteOwnerMap[pos] = (float)rand.nextGaussian() * 0.20f;
            }
          }
          resultBuf->result->whiteOwnerMap = whiteOwnerMap;
        }
        else {
          resultBuf->result->whiteOwnerMap = NULL;
        }

        //These aren't really probabilities. Win/Loss/NoResult will get softmaxed later
        double whiteWinProb = 0.0 + rand.nextGaussian() * 0.20;
        double whiteLossProb = 0.0 + rand.nextGaussian() * 0.20;
        double whiteScoreMean = 0.0 + rand.nextGaussian() * 0.20;
        double whiteScoreMeanSq = 0.0 + rand.nextGaussian() * 0.20;
        double whiteNoResultProb = 0.0 + rand.nextGaussian() * 0.20;
        double varTimeLeft = 0.5 * boardXSize * boardYSize;
        resultBuf->result->whiteWinProb = (float)whiteWinProb;
        resultBuf->result->whiteLossProb = (float)whiteLossProb;
        resultBuf->result->whiteNoResultProb = (float)whiteNoResultProb;
        resultBuf->result->whiteScoreMean = (float)whiteScoreMean;
        resultBuf->result->whiteScoreMeanSq = (float)whiteScoreMeanSq;
        resultBuf->result->whiteLead = (float)whiteScoreMean;
        resultBuf->result->varTimeLeft = (float)varTimeLeft;
        resultBuf->result->shorttermWinlossError = 0.0f;
        resultBuf->result->shorttermScoreError = 0.0f;
        resultBuf->result->policyOptimismUsed = (float)resultBuf->policyOptimism;
        resultBuf->hasResult = true;
        resultBuf->clientWaitingForResult.notify_all();
        resultLock.unlock();
      }
      m_numRowsProcessed.fetch_add(numRows, std::memory_order_relaxed);
      m_numBatchesProcessed.fetch_add(1, std::memory_order_relaxed);
      numRowsHandledThisThread += numRows;
      numBatchesHandledThisThread += 1;

      //numOngoingEvals is decremented by the client in finishOngoingEval()
      //after postprocess + cache write, so reload's drain blocks on the
      //full evaluate() lifetime — not just batch completion. The notify
      //below is a historical wakeup edge for waitForNextNNEvalIfAny()
      //condvar waiters; it is a no-op now that the API is sleep-based
      //(numWaitingEvals stays at 0), but is kept as defence in depth.
      lock.lock();
      if(numWaitingEvals > 0)
        waitingForFinish.notify_all();
      lock.unlock();
      continue; // debug path: skip profiling
    }
    else {
      outputBuf.clear();
      for(int row = 0; row<numRows; row++) {
        NNOutput* emptyOutput = nnOutputPool->acquire();
        assert(resultBufs[row] != NULL);
        emptyOutput->nnXLen = nnXLen;
        emptyOutput->nnYLen = nnYLen;
        if(resultBufs[row]->includeOwnerMap)
          emptyOutput->whiteOwnerMap = new float[nnXLen*nnYLen];
        else
          emptyOutput->whiteOwnerMap = NULL;
        outputBuf.push_back(emptyOutput);
      }

      if(_profEnabled) _tAfterAlloc = ServeProfileStats::Clock::now();
      for(int row = 0; row<numRows; row++) {
        if(resultBufs[row]->symmetry == NNInputs::SYMMETRY_NOTSPECIFIED) {
          if(doRandomize)
            resultBufs[row]->symmetry = rand.nextUInt(SymmetryHelpers::NUM_SYMMETRIES);
          else {
            assert(defaultSymmetry >= 0 && defaultSymmetry <= SymmetryHelpers::NUM_SYMMETRIES-1);
            resultBufs[row]->symmetry = defaultSymmetry;
          }
        }
      }

      // Set up overlap callback: notify previous batch during GPU computation
      std::shared_ptr<NNOutputPool> pool = nnOutputPool;
      if(prevOverlapNumRows > 0) {
        auto pBufs = std::move(prevOverlapResultBufs);
        auto pOuts = std::move(prevOverlapOutputBuf);
        int pN = prevOverlapNumRows;
        prevOverlapNumRows = 0;
        g_gpuOverlapCallback = [pBufs = std::move(pBufs), pOuts = std::move(pOuts), pN, pool]() mutable {
          for(int row = 0; row < pN; row++) {
            NNResultBuf* rb = pBufs[row];
            auto rp = std::shared_ptr<NNOutput>(pOuts[row], [pool](NNOutput* p) { pool->release(p); });
            {
              lock_guard<std::mutex> rl(rb->resultMutex);
              rb->result = std::move(rp);
              rb->hasResult = true;
            }
            rb->clientWaitingForResult.notify_one();
          }
        };
      }

      if(_profEnabled) _tBeforeGO = ServeProfileStats::Clock::now();
      NeuralNet::getOutput(gpuHandle, buf.inputBuffers, numRows, resultBufs.data(), outputBuf);
      if(_profEnabled) _tAfterGO = ServeProfileStats::Clock::now();
      assert(outputBuf.size() == numRows);

      // If backend didn't execute overlap callback (CUDA, or profiling mode), do it now
      if(g_gpuOverlapCallback) {
        g_gpuOverlapCallback();
        g_gpuOverlapCallback = nullptr;
      }

      m_numRowsProcessed.fetch_add(numRows, std::memory_order_relaxed);
      m_numBatchesProcessed.fetch_add(1, std::memory_order_relaxed);
      numRowsHandledThisThread += numRows;
      numBatchesHandledThisThread += 1;

      // Save current batch for overlap notify in next iteration
      if(_profEnabled) _tBeforeNotify = ServeProfileStats::Clock::now();
      prevOverlapResultBufs.resize(numRows);
      prevOverlapOutputBuf.resize(numRows);
      for(int row = 0; row < numRows; row++) {
        prevOverlapResultBufs[row] = resultBufs[row];
        prevOverlapOutputBuf[row] = outputBuf[row];
        resultBufs[row] = NULL;
      }
      prevOverlapNumRows = numRows;
    }

    if(_profEnabled) _tAfterNotify = ServeProfileStats::Clock::now();
    //Lock and update stats before looping again. numOngoingEvals is decremented
    //by the client in finishOngoingEval() after postprocess + cache write —
    //leaving it on the server here would let reload clear the cache between
    //our notify and the client's nnCacheTable->set, polluting the new model's
    //cache with old-weight outputs. The notify below is a historical wakeup
    //for waitForNextNNEvalIfAny() condvar waiters; it is a no-op now that
    //the API is sleep-based (numWaitingEvals stays at 0), but is kept as
    //defence in depth in case a future change brings that API back.
    lock.lock();
    if(numWaitingEvals > 0)
      waitingForFinish.notify_all();
    lock.unlock();
    if(_profEnabled) {
      auto _tEnd = ServeProfileStats::Clock::now();
      _prof.queue_pop_us  += ServeProfileStats::us(_tCycleStart, _tAfterPop);
      _prof.alloc_us      += ServeProfileStats::us(_tAfterPop, _tAfterAlloc);
      _prof.sym_us        += ServeProfileStats::us(_tAfterAlloc, _tBeforeGO);
      _prof.get_output_us += ServeProfileStats::us(_tBeforeGO, _tAfterGO);
      _prof.notify_us     += ServeProfileStats::us(_tBeforeNotify, _tAfterNotify);
      _prof.bufupd_us     += ServeProfileStats::us(_tAfterNotify, _tEnd);
      _prof.addBatch(numRows);
      if(_prof.due()) _prof.report(gpuIdxForThisThread);
    }
    continue;
  }

  NeuralNet::freeComputeHandle(gpuHandle);
  if(logger != NULL) {
    logger->write(
      "GPU " + Global::intToString(gpuIdxForThisThread) + " finishing, processed " +
      Global::int64ToString(numRowsHandledThisThread) + " rows " +
      Global::int64ToString(numBatchesHandledThisThread) + " batches"
    );
  }
}

void NNEvaluator::waitForNextNNEvalIfAny() {
  //Throttle-only API. The single caller (search.cpp's MCTS terminal-leaf
  //branch) uses this to keep a leaf-only playout from running much faster
  //than a real NN eval and skewing MCTS visit statistics. It does NOT need
  //to observe any specific numOngoingEvals transition — "wait roughly one
  //eval's worth of wall time" is the contract.
  //
  //The previous condvar implementation (++numWaitingEvals then park on
  //waitingForFinish until numOngoingEvals strictly decreased) was correct
  //but caused a 400-thread thundering herd on every evaluate() completion
  //in RL selfplay: finishOngoingEval()'s notify_all would wake every parked
  //game thread, each of which re-acquired bufferMutex only to re-sleep
  //because its initialOngoing snapshot still wasn't beaten. Net effect:
  //~2.4x throughput regression.
  //
  //A plain sleep matches the throttle contract, generates no contention on
  //bufferMutex, and keeps numWaitingEvals at 0 so finishOngoingEval()'s
  //gated notify_all becomes a no-op on the hot path. The drain path in
  //killServerThreads() still uses waitingForFinish via drainPending, so
  //shutdown semantics are unchanged.
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
}


static double softPlus(double x) {
  //Avoid blowup
  if(x > 40.0)
    return x;
  else
    return log(1.0 + exp(x));
}

static const int daggerPattern[9][8] = {
  {0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,0,2,1,0,0,0,0},
  {0,0,2,1,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,2,1,0,0,0,0,0},
  {0,3,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
};
static bool daggerMatch(const Board& board, Player nextPla, Loc& banned, int symmetry) {
  for(int yi = 0; yi < 9; yi++) {
    for(int xi = 0; xi < 8; xi++) {
      int y = yi;
      int x = xi;
      if((symmetry & 0x1) != 0)
        std::swap(x,y);
      if((symmetry & 0x2) != 0)
        x = board.x_size-1-x;
      if((symmetry & 0x4) != 0)
        y = board.y_size-1-y;
      Loc loc = Location::getLoc(x,y,board.x_size);
      int m = daggerPattern[yi][xi];
      if(m == 0 && board.colors[loc] != C_EMPTY)
        return false;
      if(m == 1 && board.colors[loc] != nextPla)
        return false;
      if(m == 2 && board.colors[loc] != getOpp(nextPla))
        return false;
      if(m == 3)
        banned = loc;
    }
  }
  return true;
}

std::shared_ptr<NNOutput>* NNEvaluator::averageMultipleSymmetries(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const SGFMetadata* sgfMeta,
  const MiscNNInputParams& baseNNInputParams,
  NNResultBuf& buf,
  bool includeOwnerMap,
  Rand& rand,
  int numSymmetriesToSample
) {
  MiscNNInputParams nnInputParams = baseNNInputParams;
  vector<std::shared_ptr<NNOutput>> ptrs;
  std::array<int, SymmetryHelpers::NUM_SYMMETRIES> symmetryIndexes;
  std::iota(symmetryIndexes.begin(), symmetryIndexes.end(), 0);
  for(int i = 0; i<numSymmetriesToSample; i++) {
    std::swap(symmetryIndexes[i], symmetryIndexes[rand.nextInt(i,SymmetryHelpers::NUM_SYMMETRIES-1)]);
    nnInputParams.symmetry = symmetryIndexes[i];
    bool skipCacheThisIteration = true; //Skip cache since there's no guarantee which symmetry is in the cache
    evaluate(
      board, history, nextPlayer, sgfMeta,
      nnInputParams,
      buf, skipCacheThisIteration, includeOwnerMap
    );
    ptrs.push_back(std::move(buf.result));
  }
  return new std::shared_ptr<NNOutput>(new NNOutput(ptrs));
}

void NNEvaluator::evaluate(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  NNResultBuf& buf,
  bool skipCache,
  bool includeOwnerMap
) {
  evaluate(
    board,
    history,
    nextPlayer,
    NULL,
    nnInputParams,
    buf,
    skipCache,
    includeOwnerMap
  );
}

void NNEvaluator::evaluate(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const SGFMetadata* sgfMeta,
  const MiscNNInputParams& nnInputParamsArg,
  NNResultBuf& buf,
  bool skipCache,
  bool includeOwnerMap
) {
  buf.hasResult = false;

  if(board.x_size > nnXLen || board.y_size > nnYLen)
    throw StringError("NNEvaluator was configured with nnXLen = " + Global::intToString(nnXLen) +
                      " nnYLen = " + Global::intToString(nnYLen) +
                      " but was asked to evaluate board with larger x or y size");
  if(requireExactNNLen) {
    if(board.x_size != nnXLen || board.y_size != nnYLen)
      throw StringError("NNEvaluator was configured with nnXLen = " + Global::intToString(nnXLen) +
                        " nnYLen = " + Global::intToString(nnYLen) +
                        " and requireExactNNLen, but was asked to evaluate board with different x or y size");
  }

  if(isSuperseded.load(std::memory_order_acquire)) {
    //Stamp the same nnHash a real-path NNOutput would carry, so that
    //averageMultipleSymmetries() (NNOutput's vector ctor) can mix dummy and
    //real results from the same query without tripping its same-hash assert.
    //numInputMetaChannels is fixed across reloads of the same model series,
    //so we can read it under a brief lock and the value matches what the
    //real path would snapshot below.
    int dummyNumMetaChannels;
    {
      std::lock_guard<std::mutex> lock(bufferMutex);
      dummyNumMetaChannels = numInputMetaChannels;
    }
    MiscNNInputParams dummyParams = nnInputParamsArg;
    if(dummyNumMetaChannels > 0)
      dummyParams.policyOptimism = 0.0;
    Hash128 dummyHash = NNInputs::getHash(board, history, nextPlayer, dummyParams);
    if(dummyNumMetaChannels > 0 && sgfMeta != NULL && sgfMeta->initialized)
      dummyHash ^= sgfMeta->getHash(nextPlayer);
    fillDummyResult(
      board, history, nextPlayer,
      nnXLen, nnYLen, policySize,
      nnInputParamsArg.policyOptimism,
      includeOwnerMap, buf
    );
    buf.result->nnHash = dummyHash;
    return;
  }

  //Do not assert(!isKilled) here: killServerThreads() toggles isKilled inside
  //the reload path in parallel with evaluate(). The forcePush-failure fallback
  //below is the authoritative handling for that race.

  //Snapshot every field that reloadLoadedModel() can replace, and claim a
  //numOngoingEvals slot, all atomically under bufferMutex. reloadLoadedModel()
  //drains numOngoingEvals == 0 and replaces these fields under the same mutex,
  //so once this block succeeds we are guaranteed a consistent view of a single
  //model for the entire rest of evaluate() (input fill + server query +
  //postprocess), even if reload replaces the fields while our request is
  //in flight on the server side.
  int snapshotModelVersion;
  int snapshotInputsVersion;
  int snapshotNumInputMetaChannels;
  ModelPostProcessParams snapshotPostProcessParams;
  string snapshotModelName;
  string snapshotModelFileName;
  {
    unique_lock<std::mutex> lock(bufferMutex);
    if(isSuperseded.load(std::memory_order_acquire)) {
      int dummyNumMetaChannels = numInputMetaChannels;
      lock.unlock();
      MiscNNInputParams dummyParams = nnInputParamsArg;
      if(dummyNumMetaChannels > 0)
        dummyParams.policyOptimism = 0.0;
      Hash128 dummyHash = NNInputs::getHash(board, history, nextPlayer, dummyParams);
      if(dummyNumMetaChannels > 0 && sgfMeta != NULL && sgfMeta->initialized)
        dummyHash ^= sgfMeta->getHash(nextPlayer);
      fillDummyResult(
        board, history, nextPlayer,
        nnXLen, nnYLen, policySize,
        nnInputParamsArg.policyOptimism,
        includeOwnerMap, buf
      );
      buf.result->nnHash = dummyHash;
      return;
    }
    snapshotModelVersion = modelVersion;
    snapshotInputsVersion = inputsVersion;
    snapshotNumInputMetaChannels = numInputMetaChannels;
    snapshotPostProcessParams = postProcessParams;
    snapshotModelName = modelName;
    snapshotModelFileName = modelFileName;
    numOngoingEvals += 1;
  }
  //RAII guard balances the numOngoingEvals++ above on every exit path
  //(cache hit early return, forcePush-fail fallback, normal completion
  //after postprocess + cache write, or any exception thrown during
  //postprocess). reloadLoadedModel()'s drain loop waits on
  //numOngoingEvals == 0 before it mutates snapshotted fields or clears
  //the cache, so keeping the slot alive across the full evaluate()
  //lifetime prevents two hazards in one stroke: postprocess reading the
  //replaced params, and nnCacheTable->set() inserting an old-weight
  //NNOutput into the cleared-for-new-model cache.
  Global::CustomScopeGuard evalSlotGuard([this]() { finishOngoingEval(); });

  // Avoid using policy optimism for humanSL
  MiscNNInputParams nnInputParams = nnInputParamsArg;
  if(snapshotNumInputMetaChannels > 0)
    nnInputParams.policyOptimism = 0.0;

  Hash128 nnHash = NNInputs::getHash(board, history, nextPlayer, nnInputParams);
  if(snapshotNumInputMetaChannels > 0) {
    if(sgfMeta == NULL)
      Global::fatalError("SGFMetadata is required for " + snapshotModelName + " but was not provided");
    if(!sgfMeta->initialized)
      Global::fatalError("SGFMetadata is required for " + snapshotModelName + " but was not initialized. Did you specify humanSLProfile=... in katago's config or via overrides?");
    nnHash ^= sgfMeta->getHash(nextPlayer);
  }

  bool hadResultWithoutOwnerMap = false;
  shared_ptr<NNOutput> resultWithoutOwnerMap;
  if(nnCacheTable != NULL && !skipCache && nnCacheTable->get(nnHash,buf.result)) {
    if(!(includeOwnerMap && buf.result->whiteOwnerMap == NULL))
    {
      buf.hasResult = true;
      //evalSlotGuard releases the numOngoingEvals slot on return.
      return;
    }
    else {
      hadResultWithoutOwnerMap = true;
      resultWithoutOwnerMap = std::move(buf.result);
      buf.result = nullptr;
    }
  }
  buf.includeOwnerMap = includeOwnerMap;

  buf.boardXSizeForServer = board.x_size;
  buf.boardYSizeForServer = board.y_size;

  if(!debugSkipNeuralNet) {
    const int rowSpatialLen = NNModelVersion::getNumSpatialFeatures(snapshotModelVersion) * nnXLen * nnYLen;
    if(buf.rowSpatialBuf.size() < rowSpatialLen)
      buf.rowSpatialBuf.resize(rowSpatialLen);
    const int rowGlobalLen = NNModelVersion::getNumGlobalFeatures(snapshotModelVersion);
    if(buf.rowGlobalBuf.size() < rowGlobalLen)
      buf.rowGlobalBuf.resize(rowGlobalLen);
    const int rowMetaLen = snapshotNumInputMetaChannels;
    if(buf.rowMetaBuf.size() < rowMetaLen)
      buf.rowMetaBuf.resize(rowMetaLen);

    static_assert(NNModelVersion::latestInputsVersionImplemented == 7, "");
    if(snapshotInputsVersion == 3)
      NNInputs::fillRowV3(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
    else if(snapshotInputsVersion == 4)
      NNInputs::fillRowV4(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
    else if(snapshotInputsVersion == 5)
      NNInputs::fillRowV5(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
    else if(snapshotInputsVersion == 6)
      NNInputs::fillRowV6(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
    else if(snapshotInputsVersion == 7)
      NNInputs::fillRowV7(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
    else
      ASSERT_UNREACHABLE;

    if(rowMetaLen > 0) {
      if(sgfMeta == NULL)
        Global::fatalError("SGFMetadata is required for " + snapshotModelName + " but was not provided");
      if(!sgfMeta->initialized)
        Global::fatalError("SGFMetadata is required for " + snapshotModelName + " but was not initialized. Did you specify humanSLProfile=... in katago's config or via overrides?");
      SGFMetadata::fillMetadataRow(
        sgfMeta,
        buf.rowMetaBuf.data(),
        nextPlayer,
        board.x_size*board.y_size
      );
      buf.hasRowMeta = true;
    }
    else {
      buf.hasRowMeta = false;
    }
  }

  buf.symmetry = nnInputParams.symmetry;
  buf.policyOptimism = nnInputParams.policyOptimism;

  bool suc = queryQueue.forcePush(&buf);
  if(!suc) {
    //The query queue was set readonly by reloadLoadedModel() (or destructor)
    //between our isSuperseded check and this push. Fall back to a dummy
    //result — without this the thread would block forever on
    //buf.clientWaitingForResult since no server will process a buf that
    //was never pushed. Stamp the same nnHash we already computed for the
    //real path so a sibling symmetry can still average against this dummy
    //without tripping NNOutput's same-hash assert. evalSlotGuard releases
    //the numOngoingEvals slot on return, waking the reload drain loop.
    fillDummyResult(
      board, history, nextPlayer,
      nnXLen, nnYLen, policySize,
      nnInputParams.policyOptimism,
      includeOwnerMap, buf
    );
    buf.result->nnHash = nnHash;
    return;
  }

  unique_lock<std::mutex> resultLock(buf.resultMutex);
  while(!buf.hasResult)
    buf.clientWaitingForResult.wait(resultLock);
  resultLock.unlock();

  //Perform postprocessing on the result - turn the nn output into probabilities
  //As a hack though, if the only thing we were missing was the ownermap, just grab the old policy and values
  //and use those. This avoids recomputing in a randomly different orientation when we just need the ownermap
  //and causing policy weights to be different, which would reduce performance of successive searches in a game
  //by making the successive searches distribute their playouts less coherently and using the cache more poorly.
  if(hadResultWithoutOwnerMap) {
    buf.result->whiteWinProb = resultWithoutOwnerMap->whiteWinProb;
    buf.result->whiteLossProb = resultWithoutOwnerMap->whiteLossProb;
    buf.result->whiteNoResultProb = resultWithoutOwnerMap->whiteNoResultProb;
    buf.result->whiteScoreMean = resultWithoutOwnerMap->whiteScoreMean;
    buf.result->whiteScoreMeanSq = resultWithoutOwnerMap->whiteScoreMeanSq;
    buf.result->whiteLead = resultWithoutOwnerMap->whiteLead;
    buf.result->varTimeLeft = resultWithoutOwnerMap->varTimeLeft;
    buf.result->shorttermWinlossError = resultWithoutOwnerMap->shorttermWinlossError;
    buf.result->shorttermScoreError = resultWithoutOwnerMap->shorttermScoreError;
    std::copy(resultWithoutOwnerMap->policyProbs, resultWithoutOwnerMap->policyProbs + NNPos::MAX_NN_POLICY_SIZE, buf.result->policyProbs);
    buf.result->policyOptimismUsed = (float)resultWithoutOwnerMap->policyOptimismUsed;
    buf.result->nnXLen = resultWithoutOwnerMap->nnXLen;
    buf.result->nnYLen = resultWithoutOwnerMap->nnYLen;
    assert(buf.result->whiteOwnerMap != NULL);
  }
  else {
    float* policy = buf.result->policyProbs;

    float policyOutputScaling = snapshotPostProcessParams.outputScaleMultiplier / nnInputParams.nnPolicyTemperature;

    int xSize = board.x_size;
    int ySize = board.y_size;

    float maxPolicy = -1e25f;
    bool isLegal[NNPos::MAX_NN_POLICY_SIZE];
    int legalCount = 0;
    assert(nextPlayer == history.presumedNextMovePla);
    for(int i = 0; i<policySize; i++) {
      Loc loc = NNPos::posToLoc(i,xSize,ySize,nnXLen,nnYLen);
      isLegal[i] = history.isLegal(board,loc,nextPlayer);
    }

    if(nnInputParams.avoidMYTDaggerHack && xSize >= 13 && ySize >= 13) {
      for(int symmetry = 0; symmetry < 8; symmetry++) {
        Loc banned = Board::NULL_LOC;
        if(daggerMatch(board, nextPlayer, banned, symmetry)) {
          if(banned != Board::NULL_LOC) {
            isLegal[NNPos::locToPos(banned,xSize,nnXLen,nnYLen)] = false;
          }
        }
      }
    }

    for(int i = 0; i<policySize; i++) {
      float policyValue;
      if(isLegal[i]) {
        legalCount += 1;
        policyValue = policy[i] * policyOutputScaling;
      }
      else
        policyValue = -1e30f;

      policy[i] = policyValue;
      if(policyValue > maxPolicy)
        maxPolicy = policyValue;
    }

    assert(legalCount > 0);

    float policySum = 0.0f;

    if(nnInputParams.enablePassingHacks) {
      //Cap passing prior policy at 95% (19x other moves)
      float maxPassPolicySumFactor = 19.0f;

      for(int i = 0; i<policySize-1; i++) {
        policy[i] = exp(policy[i] - maxPolicy);
        policySum += policy[i];
      }
      int passPos = NNPos::locToPos(Board::PASS_LOC, xSize, nnXLen, nnYLen);
      assert(passPos == policySize-1);
      int i = passPos;
      policy[i] = std::max(1e-20f, std::min(exp(policy[i] - maxPolicy), policySum * maxPassPolicySumFactor));
      policySum += policy[i];
    }
    else {
      for(int i = 0; i<policySize; i++) {
        policy[i] = exp(policy[i] - maxPolicy);
        policySum += policy[i];
      }
    }

    if(!isfinite(policySum)) {
      cout << "Got nonfinite for policy sum" << endl;
      history.printDebugInfo(cout,board);
      throw StringError("Got nonfinite for policy sum");
    }

    //Somehow all legal moves rounded to 0 probability
    if(policySum <= 0.0) {
      if(!buf.errorLogLockout && logger != NULL) {
        buf.errorLogLockout = true;
        logger->write("Warning: all legal moves rounded to 0 probability for " + snapshotModelFileName);
      }
      float uniform = 1.0f / legalCount;
      for(int i = 0; i<policySize; i++) {
        policy[i] = isLegal[i] ? uniform : -1.0f;
      }
    }
    //Normal case
    else {
      for(int i = 0; i<policySize; i++)
        policy[i] = isLegal[i] ? (policy[i] / policySum) : -1.0f;
    }

    //Fill everything out-of-bounds too, for robustness.
    for(int i = policySize; i<NNPos::MAX_NN_POLICY_SIZE; i++)
      policy[i] = -1.0f;

    buf.result->policyOptimismUsed = (float)nnInputParams.policyOptimism;

    //Fix up the value as well. Note that the neural net gives us back the value from the perspective
    //of the player so we need to negate that to make it the white value.
    if(snapshotModelVersion == 3) {
      const double twoOverPi = 0.63661977236758134308;

      double winProb;
      double lossProb;
      double noResultProb;
      //Version 3 neural nets just pack the pre-arctanned scoreValue into the whiteScoreMean field
      double scoreValue = atan(buf.result->whiteScoreMean * snapshotPostProcessParams.outputScaleMultiplier) * twoOverPi;
      {
        double winLogits = buf.result->whiteWinProb * snapshotPostProcessParams.outputScaleMultiplier;
        double lossLogits = buf.result->whiteLossProb * snapshotPostProcessParams.outputScaleMultiplier;
        double noResultLogits = buf.result->whiteNoResultProb * snapshotPostProcessParams.outputScaleMultiplier;

        //Softmax
        double maxLogits = std::max(std::max(winLogits,lossLogits),noResultLogits);
        winProb = exp(winLogits - maxLogits);
        lossProb = exp(lossLogits - maxLogits);
        noResultProb = exp(noResultLogits - maxLogits);

        double probSum = winProb + lossProb + noResultProb;
        winProb /= probSum;
        lossProb /= probSum;
        noResultProb /= probSum;

        if(!isfinite(probSum) || !isfinite(scoreValue)) {
          cout << "Got nonfinite for nneval value" << endl;
          cout << winLogits << " " << lossLogits << " " << noResultLogits << " " << scoreValue << endl;
          throw StringError("Got nonfinite for nneval value");
        }
      }

      if(nextPlayer == P_WHITE) {
        buf.result->whiteWinProb = (float)winProb;
        buf.result->whiteLossProb = (float)lossProb;
        buf.result->whiteNoResultProb = (float)noResultProb;
        buf.result->whiteScoreMean = (float)ScoreValue::approxWhiteScoreOfScoreValueSmooth(scoreValue,0.0,2.0,board.sqrtBoardArea());
        buf.result->whiteScoreMeanSq = buf.result->whiteScoreMean * buf.result->whiteScoreMean;
        buf.result->whiteLead = buf.result->whiteScoreMean;
        buf.result->varTimeLeft = -1;
        buf.result->shorttermWinlossError = -1;
        buf.result->shorttermScoreError = -1;
      }
      else {
        buf.result->whiteWinProb = (float)lossProb;
        buf.result->whiteLossProb = (float)winProb;
        buf.result->whiteNoResultProb = (float)noResultProb;
        buf.result->whiteScoreMean = -(float)ScoreValue::approxWhiteScoreOfScoreValueSmooth(scoreValue,0.0,2.0,board.sqrtBoardArea());
        buf.result->whiteScoreMeanSq = buf.result->whiteScoreMean * buf.result->whiteScoreMean;
        buf.result->whiteLead = buf.result->whiteScoreMean;
        buf.result->varTimeLeft = -1;
        buf.result->shorttermWinlossError = -1;
        buf.result->shorttermScoreError = -1;
      }

    }
    else if(snapshotModelVersion >= 4) {
      double winProb;
      double lossProb;
      double noResultProb;
      double scoreMean;
      double scoreMeanSq;
      double lead;
      double varTimeLeft;
      double shorttermWinlossError;
      double shorttermScoreError;
      {
        double winLogits = buf.result->whiteWinProb * snapshotPostProcessParams.outputScaleMultiplier;
        double lossLogits = buf.result->whiteLossProb * snapshotPostProcessParams.outputScaleMultiplier;
        double noResultLogits = buf.result->whiteNoResultProb * snapshotPostProcessParams.outputScaleMultiplier;
        double scoreMeanPreScaled = buf.result->whiteScoreMean * snapshotPostProcessParams.outputScaleMultiplier;
        double scoreStdevPreSoftplus = buf.result->whiteScoreMeanSq * snapshotPostProcessParams.outputScaleMultiplier;
        double leadPreScaled = buf.result->whiteLead * snapshotPostProcessParams.outputScaleMultiplier;
        double varTimeLeftPreSoftplus = buf.result->varTimeLeft * snapshotPostProcessParams.outputScaleMultiplier;
        double shorttermWinlossErrorPreSoftplus = buf.result->shorttermWinlossError * snapshotPostProcessParams.outputScaleMultiplier;
        double shorttermScoreErrorPreSoftplus = buf.result->shorttermScoreError * snapshotPostProcessParams.outputScaleMultiplier;

        if(history.rules.koRule != Rules::KO_SIMPLE && history.rules.scoringRule != Rules::SCORING_TERRITORY)
          noResultLogits -= 100000.0;

        //Softmax
        double maxLogits = std::max(std::max(winLogits,lossLogits),noResultLogits);
        winProb = exp(winLogits - maxLogits);
        lossProb = exp(lossLogits - maxLogits);
        noResultProb = exp(noResultLogits - maxLogits);

        if(history.rules.koRule != Rules::KO_SIMPLE && history.rules.scoringRule != Rules::SCORING_TERRITORY)
          noResultProb = 0.0;

        double probSum = winProb + lossProb + noResultProb;
        winProb /= probSum;
        lossProb /= probSum;
        noResultProb /= probSum;

        scoreMean = scoreMeanPreScaled * snapshotPostProcessParams.scoreMeanMultiplier;
        double scoreStdev = softPlus(scoreStdevPreSoftplus) * snapshotPostProcessParams.scoreStdevMultiplier;
        scoreMeanSq = scoreMean * scoreMean + scoreStdev * scoreStdev;
        lead = leadPreScaled * snapshotPostProcessParams.leadMultiplier;
        varTimeLeft = softPlus(varTimeLeftPreSoftplus) * snapshotPostProcessParams.varianceTimeMultiplier;

        //scoreMean and scoreMeanSq are still conditional on having a result, we need to make them unconditional now
        //noResult counts as 0 score for scorevalue purposes.
        scoreMean = scoreMean * (1.0-noResultProb);
        scoreMeanSq = scoreMeanSq * (1.0-noResultProb);
        lead = lead * (1.0-noResultProb);

        if(snapshotModelVersion >= 14) {
          {
            double s = softPlus(shorttermWinlossErrorPreSoftplus * 0.5);
            shorttermWinlossError = sqrt(s * s * snapshotPostProcessParams.shorttermValueErrorMultiplier);
          }
          {
            double s = softPlus(shorttermScoreErrorPreSoftplus * 0.5);
            shorttermScoreError = sqrt(s * s * snapshotPostProcessParams.shorttermScoreErrorMultiplier);
          }
        }
        else if(snapshotModelVersion >= 10) {
          shorttermWinlossError = sqrt(softPlus(shorttermWinlossErrorPreSoftplus) * snapshotPostProcessParams.shorttermValueErrorMultiplier);
          shorttermScoreError = sqrt(softPlus(shorttermScoreErrorPreSoftplus) * snapshotPostProcessParams.shorttermScoreErrorMultiplier);
        }
        else {
          shorttermWinlossError = softPlus(shorttermWinlossErrorPreSoftplus);
          shorttermScoreError = softPlus(shorttermScoreErrorPreSoftplus) * 10.0;
        }

        if(
          !isfinite(probSum) ||
          !isfinite(scoreMean) ||
          !isfinite(scoreMeanSq) ||
          !isfinite(lead) ||
          !isfinite(varTimeLeft) ||
          !isfinite(shorttermWinlossError) ||
          !isfinite(shorttermScoreError)
        ) {
          cout << "Got nonfinite for nneval value" << endl;
          cout << winLogits << " " << lossLogits << " " << noResultLogits
               << " " << scoreMean << " " << scoreMeanSq
               << " " << lead << " " << varTimeLeft
               << " " << shorttermWinlossError << " " << shorttermScoreError
               << endl;
          throw StringError("Got nonfinite for nneval value");
        }
      }

      if(nextPlayer == P_WHITE) {
        buf.result->whiteWinProb = (float)winProb;
        buf.result->whiteLossProb = (float)lossProb;
        buf.result->whiteNoResultProb = (float)noResultProb;
        buf.result->whiteScoreMean = (float)scoreMean;
        buf.result->whiteScoreMeanSq = (float)scoreMeanSq;
        buf.result->whiteLead = (float)lead;
      }
      else {
        buf.result->whiteWinProb = (float)lossProb;
        buf.result->whiteLossProb = (float)winProb;
        buf.result->whiteNoResultProb = (float)noResultProb;
        buf.result->whiteScoreMean = -(float)scoreMean;
        buf.result->whiteScoreMeanSq = (float)scoreMeanSq;
        buf.result->whiteLead = -(float)lead;
      }

      if(snapshotModelVersion >= 9) {
        buf.result->varTimeLeft = (float)varTimeLeft;
        buf.result->shorttermWinlossError = (float)shorttermWinlossError;
        buf.result->shorttermScoreError = (float)shorttermScoreError;
      }
      else {
        buf.result->varTimeLeft = -1;
        buf.result->shorttermWinlossError = -1;
        buf.result->shorttermScoreError = -1;
      }
    }
    else {
      throw StringError("NNEval value postprocessing not implemented for model version");
    }
  }

  //Postprocess ownermap
  if(buf.result->whiteOwnerMap != NULL) {
    if(snapshotModelVersion >= 3) {
      for(int pos = 0; pos<nnXLen*nnYLen; pos++) {
        int y = pos / nnXLen;
        int x = pos % nnXLen;
        if(y >= board.y_size || x >= board.x_size)
          buf.result->whiteOwnerMap[pos] = 0.0f;
        else {
          //Similarly as mentioned above, the result we get back from the net is actually not from white's perspective,
          //but from the player to move, so we need to flip it to make it white at the same time as we tanh it.
          if(nextPlayer == P_WHITE)
            buf.result->whiteOwnerMap[pos] = tanh(buf.result->whiteOwnerMap[pos] * snapshotPostProcessParams.outputScaleMultiplier);
          else
            buf.result->whiteOwnerMap[pos] = -tanh(buf.result->whiteOwnerMap[pos] * snapshotPostProcessParams.outputScaleMultiplier);
        }
      }
    }
    else {
      throw StringError("NNEval value postprocessing not implemented for model version");
    }
  }


  //And record the nnHash in the result and put it into the table
  buf.result->nnHash = nnHash;
  if(nnCacheTable != NULL)
    nnCacheTable->set(buf.result);

}

//Uncomment this to lower the effective hash size down to one where we get true collisions
//#define SIMULATE_TRUE_HASH_COLLISIONS

NNCacheTable::Entry::Entry()
  :ptr(nullptr)
{}
NNCacheTable::Entry::~Entry()
{}

NNCacheTable::NNCacheTable(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo) {
  if(sizePowerOfTwo < 0 || sizePowerOfTwo > 63)
    throw StringError("NNCacheTable: Invalid sizePowerOfTwo: " + Global::intToString(sizePowerOfTwo));
  if(mutexPoolSizePowerOfTwo < 0 || mutexPoolSizePowerOfTwo > 31)
    throw StringError("NNCacheTable: Invalid mutexPoolSizePowerOfTwo: " + Global::intToString(mutexPoolSizePowerOfTwo));
#if defined(SIMULATE_TRUE_HASH_COLLISIONS)
  sizePowerOfTwo = sizePowerOfTwo > 12 ? 12 : sizePowerOfTwo;
#endif
  if(mutexPoolSizePowerOfTwo > sizePowerOfTwo)
    mutexPoolSizePowerOfTwo = sizePowerOfTwo;

  tableSize = ((uint64_t)1) << sizePowerOfTwo;
  tableMask = tableSize-1;
  entries = new Entry[tableSize];
  uint32_t mutexPoolSize = ((uint32_t)1) << mutexPoolSizePowerOfTwo;
  mutexPoolMask = mutexPoolSize-1;
  mutexPool = new MutexPool(mutexPoolSize);
}
NNCacheTable::~NNCacheTable() {
  delete[] entries;
  delete mutexPool;
}

bool NNCacheTable::get(Hash128 nnHash, shared_ptr<NNOutput>& ret) {
  //Free ret BEFORE locking, to avoid any expensive operations while locked.
  if(ret != nullptr)
    ret.reset();

  uint64_t idx = nnHash.hash0 & tableMask;
  uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
  Entry& entry = entries[idx];
  std::mutex& mutex = mutexPool->getMutex(mutexIdx);

  std::lock_guard<std::mutex> lock(mutex);

  bool found = false;
#if defined(SIMULATE_TRUE_HASH_COLLISIONS)
  if(entry.ptr != nullptr && ((entry.ptr->nnHash.hash0 ^ nnHash.hash0) & 0xFFF) == 0) {
    ret = entry.ptr;
    found = true;
  }
#else
  if(entry.ptr != nullptr && entry.ptr->nnHash == nnHash) {
    ret = entry.ptr;
    found = true;
  }
#endif
  return found;
}

void NNCacheTable::set(const shared_ptr<NNOutput>& p) {
  //Immediately copy p right now, before locking, to avoid any expensive operations while locked.
  shared_ptr<NNOutput> buf(p);

  uint64_t idx = p->nnHash.hash0 & tableMask;
  uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
  Entry& entry = entries[idx];
  std::mutex& mutex = mutexPool->getMutex(mutexIdx);

  {
    std::lock_guard<std::mutex> lock(mutex);
    //Perform a swap, to avoid any expensive free under the mutex.
    entry.ptr.swap(buf);
  }

  //No longer locked, allow buf to fall out of scope now, will free whatever used to be present in the table.
}

void NNCacheTable::clear() {
  shared_ptr<NNOutput> buf;
  for(size_t idx = 0; idx<tableSize; idx++) {
    Entry& entry = entries[idx];
    uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
    std::mutex& mutex = mutexPool->getMutex(mutexIdx);
    {
      std::lock_guard<std::mutex> lock(mutex);
      entry.ptr.swap(buf);
    }
    buf.reset();
  }
}
