#ifndef NEURALNET_NNEVAL_H_
#define NEURALNET_NNEVAL_H_

#include <memory>

#include "../core/global.h"
#include "../core/commontypes.h"
#include "../core/logger.h"
#include "../core/multithread.h"
#include "../core/threadsafequeue.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/sgfmetadata.h"
#include "../neuralnet/nninterface.h"
#include "../search/mutexpool.h"

class NNEvaluator;

class NNCacheTable {
  struct Entry {
    std::shared_ptr<NNOutput> ptr;
    Entry();
    ~Entry();
  };

  Entry* entries;
  MutexPool* mutexPool;
  uint64_t tableSize;
  uint64_t tableMask;
  uint32_t mutexPoolMask;

 public:
  NNCacheTable(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo);
  ~NNCacheTable();

  NNCacheTable(const NNCacheTable& other) = delete;
  NNCacheTable& operator=(const NNCacheTable& other) = delete;

  //These are thread-safe. For get, ret will be set to nullptr upon a failure to find.
  bool get(Hash128 nnHash, std::shared_ptr<NNOutput>& ret);
  void set(const std::shared_ptr<NNOutput>& p);
  void clear();
};

//Each thread should allocate and re-use one of these
struct NNResultBuf {
  std::condition_variable clientWaitingForResult;
  std::mutex resultMutex;
  bool hasResult;
  bool includeOwnerMap;
  int boardXSizeForServer;
  int boardYSizeForServer;
  std::vector<float> rowSpatialBuf;
  std::vector<float> rowGlobalBuf;
  std::vector<float> rowMetaBuf;
  bool hasRowMeta;
  std::shared_ptr<NNOutput> result;
  bool errorLogLockout; //error flag to restrict log to 1 error to prevent spam
  int symmetry; //The symmetry to use for this eval
  double policyOptimism; //The policy optimism to use for this eval

  NNResultBuf();
  ~NNResultBuf();
  NNResultBuf(const NNResultBuf& other) = delete;
  NNResultBuf& operator=(const NNResultBuf& other) = delete;
};

//Each server thread should allocate and re-use one of these
struct NNServerBuf {
  InputBuffers* inputBuffers;

  NNServerBuf(const NNEvaluator& nneval, const LoadedModel* model);
  ~NNServerBuf();
  NNServerBuf(const NNServerBuf& other) = delete;
  NNServerBuf& operator=(const NNServerBuf& other) = delete;
};

class NNEvaluator {
 public:
  NNEvaluator(
    const std::string& modelName,
    const std::string& modelFileName,
    const std::string& expectedSha256,
    Logger* logger,
    int maxBatchSize,
    int nnXLen,
    int nnYLen,
    bool requireExactNNLen,
    bool inputsUseNHWC,
    int nnCacheSizePowerOfTwo,
    int nnMutexPoolSizePowerofTwo,
    bool debugSkipNeuralNet,
    const std::string& openCLTunerFile,
    const std::string& homeDataDirOverride,
    bool openCLReTunePerBoardSize,
    enabled_t useFP16Mode,
    enabled_t useNHWCMode,
    int numThreads,
    const std::vector<int>& gpuIdxByServerThread,
    const std::string& randSeed,
    bool doRandomize,
    int defaultSymmetry
  );
  ~NNEvaluator();

  NNEvaluator(const NNEvaluator& other) = delete;
  NNEvaluator& operator=(const NNEvaluator& other) = delete;

  std::string getModelName() const;
  std::string getModelFileName() const;
  std::string getInternalModelName() const;
  std::string getAbbrevInternalModelName() const;
  Logger* getLogger();
  bool isNeuralNetLess() const;
  int getMaxBatchSize() const;
  int getCurrentBatchSize() const;
  void setCurrentBatchSize(int batchSize);
  bool requiresSGFMetadata() const;

  int getNumGpus() const;
  int getNumServerThreads() const;
  std::set<int> getGpuIdxs() const;
  int getNNXLen() const;
  int getNNYLen() const;
  int getModelVersion() const;
  double getTrunkSpatialConvDepth() const;
  enabled_t getUsingFP16Mode() const;
  enabled_t getUsingNHWCMode() const;

  //Check if the loaded neural net supports shorttermError fields
  bool supportsShorttermError() const;

  //Return the "nearest" supported ruleset to desiredRules by this model.
  //Fills supported with true if desiredRules itself was exactly supported, false if some modifications had to be made.
  Rules getSupportedRules(const Rules& desiredRules, bool& supported);

  //Clear all entires cached in the table
  void clearCache();

  //Queue a position for the next neural net batch evaluation and wait for it. Upon evaluation, result
  //will be supplied in NNResultBuf& buf, the shared_ptr there can grabbed via std::move if desired.
  //logStream is for some error logging, can be NULL.
  //This function is threadsafe.
  void evaluate(
    const Board& board,
    const BoardHistory& history,
    Player nextPlayer,
    const MiscNNInputParams& nnInputParams,
    NNResultBuf& buf,
    bool skipCache,
    bool includeOwnerMap
  );
  void evaluate(
    const Board& board,
    const BoardHistory& history,
    Player nextPlayer,
    const SGFMetadata* sgfMeta,
    const MiscNNInputParams& nnInputParams,
    NNResultBuf& buf,
    bool skipCache,
    bool includeOwnerMap
  );
  std::shared_ptr<NNOutput>* averageMultipleSymmetries(
    const Board& board,
    const BoardHistory& history,
    Player nextPlayer,
    const SGFMetadata* sgfMeta,
    const MiscNNInputParams& baseNNInputParams,
    NNResultBuf& buf,
    bool includeOwnerMap,
    Rand& rand,
    int numSymmetriesToSample
  );

  //If there is at least one evaluate ongoing, wait until at least one finishes.
  //Returns immediately if there isn't one ongoing right now.
  void waitForNextNNEvalIfAny();

  //Actually spawn threads to handle evaluations.
  //If doRandomize, uses randSeed as a seed, further randomized per-thread
  //If not doRandomize, uses defaultSymmetry for all nn evaluations, unless a symmetry is requested in MiscNNInputParams.
  //This function itself is not threadsafe.
  void spawnServerThreads();

  //Kill spawned server threads and join and free them. This function is not threadsafe, and along with spawnServerThreads
  //should have calls to it and spawnServerThreads singlethreaded.
  void killServerThreads();

  //IN-PLACE RELOAD: swap the backing neural net model inside this evaluator without tearing down the
  //NNEvaluator object, its NNCacheTable, or its ModelData wrapper in SelfplayManager. Drains in-flight
  //evals by flipping isSuperseded during the swap so new evaluate() calls return a dummy result, then
  //kills server threads, replaces loadedModel, clears cache, respawns threads. Caller must guarantee
  //no other thread is concurrently calling spawn/kill server threads on this evaluator.
  //Returns true on success. Only meaningful for non-debugSkipNeuralNet evaluators.
  bool reloadLoadedModel(
    const std::string& newModelFileName,
    const std::string& newModelName,
    const std::string& expectedSha256
  );

  //Retire active eval service for this model but keep TRT/backend/server-thread state alive.
  //After this, future evaluate() calls fall back to a cheap dummy result so that superseded models can
  //flush out in-flight selfplay threads without retaining NN cache memory or taking new work.
  //Safe to call before destructor. Destructor is idempotent with respect to this.
  void releaseHeavyResources();
  //Free TRT/server-thread/backend state after all in-flight users are gone.
  //Safe to call multiple times and before destructor.
  void releaseRetiredBackendResources();
  //Free shared backend/model objects after all in-flight users are gone.
  //Safe to call multiple times and before destructor.
  void releaseRetiredModelResources();

  //Internal helper: balance one numOngoingEvals slot acquired in evaluate()
  //and notify killServerThreads()'s drain loop. numOngoingEvals now spans
  //the entire evaluate() lifetime (input fill + server batch + client
  //postprocess + cache write), so reloadLoadedModel() blocks on the drain
  //loop until the cache write completes — preventing an old-model NNOutput
  //from being inserted into the cache after reload's nnCacheTable->clear().
  void finishOngoingEval();
  bool isReleasedOrSuperseded() const;

  //Set the number of threads and what gpus they use. Only call this if threads are not spawned yet, or have been killed.
  void setNumThreads(const std::vector<int>& gpuIdxByServerThr);

  //After spawnServerThreads has returned, check if is was using FP16.
  bool isAnyThreadUsingFP16() const;

  //These are thread-safe. Setting them in the middle of operation might only affect future
  //neural net evals, rather than any in-flight.
  bool getDoRandomize() const;
  int getDefaultSymmetry() const;
  void setDoRandomize(bool b);
  void setDefaultSymmetry(int s);

  //Some stats
  uint64_t numRowsProcessed() const;
  uint64_t numBatchesProcessed() const;
  double averageProcessedBatchSize() const;

  void clearStats();

 private:
  std::string modelName;
  std::string modelFileName;
  const int nnXLen;
  const int nnYLen;
  const bool requireExactNNLen;
  const int policySize;
  const bool inputsUseNHWC;
  const enabled_t usingFP16Mode;
  const enabled_t usingNHWCMode;
  int numThreads;
  std::vector<int> gpuIdxByServerThread;
  const std::string randSeed;
  const bool debugSkipNeuralNet;

  ComputeContext* computeContext;
  LoadedModel* loadedModel;
  NNCacheTable* nnCacheTable;
  Logger* logger;

  std::string internalModelName;
  int modelVersion;
  int inputsVersion;
  int numInputMetaChannels;

  ModelPostProcessParams postProcessParams;

  int numServerThreadsEverSpawned;
  std::vector<std::thread*> serverThreads;

  const int maxBatchSize;

  //Counters for statistics
  std::atomic<uint64_t> m_numRowsProcessed;
  std::atomic<uint64_t> m_numBatchesProcessed;

  mutable std::mutex bufferMutex;

  //Everything in this section is protected under bufferMutex--------------------------------------------

  bool isKilled; //Flag used for killing server threads
  int numServerThreadsStartingUp; //Counter for waiting until server threads are spawned
  std::condition_variable mainThreadWaitingForSpawn; //Condvar for waiting until server threads are spawned

  std::vector<int> serverThreadsIsUsingFP16;

  int numOngoingEvals; //Current number of ongoing evals.
  int numWaitingEvals; //Legacy waiter count. Stays at 0 now that waitForNextNNEvalIfAny() is sleep-based; retained as a defensive edge for a future condvar-based throttling revival.
  int numEvalsToAwaken; //Legacy field retained for killServerThreads()'s post-drain assert. Stays at 0 under the counter-watch wakeup scheme.
  bool drainPending; //True while killServerThreads() is spinning on numOngoingEvals==0. Tells finishOngoingEval() to broadcast on waitingForFinish during shutdown drain.
  std::condition_variable waitingForFinish; //Condvar for waiting for at least one ongoing eval to finish.

  //-------------------------------------------------------------------------------------------------

  //Randomization settings for symmetries
  std::atomic<bool> currentDoRandomize;
  std::atomic<int> currentDefaultSymmetry;
  //Modifiable batch size smaller than maxBatchSize
  std::atomic<int> currentBatchSize;
  std::atomic<bool> isSuperseded;

  // Object pool for NNOutput to reduce heap allocation overhead in serve().
  // Each serve() batch allocates ~192 NNOutput objects; pooling avoids malloc/free syscalls.
  struct NNOutputPool {
    std::vector<NNOutput*> freeList;
    std::mutex mu;
    ~NNOutputPool() { for(auto* p : freeList) delete p; }
    NNOutput* acquire() {
      {
        std::lock_guard<std::mutex> lock(mu);
        if(!freeList.empty()) {
          NNOutput* p = freeList.back();
          freeList.pop_back();
          return p;
        }
      }
      return new NNOutput();
    }
    void release(NNOutput* p) {
      delete[] p->whiteOwnerMap;
      p->whiteOwnerMap = nullptr;
      delete[] p->noisedPolicyProbs;
      p->noisedPolicyProbs = nullptr;
      std::lock_guard<std::mutex> lock(mu);
      freeList.push_back(p);
    }
  };
  std::shared_ptr<NNOutputPool> nnOutputPool;

  //Queued up requests
  ThreadSafeQueue<NNResultBuf*> queryQueue;

 public:
  //Helper, for internal use only
  void serve(NNServerBuf& buf, Rand& rand, int gpuIdxForThisThread, int serverThreadIdx);
};

#endif  // NEURALNET_NNEVAL_H_
