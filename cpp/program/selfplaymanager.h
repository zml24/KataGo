#ifndef PROGRAM_SELFPLAYMANAGER_H_
#define PROGRAM_SELFPLAYMANAGER_H_

#include "../core/threadsafequeue.h"
#include "../core/timer.h"
#include "../dataio/sgf.h"
#include "../dataio/trainingwrite.h"
#include "../neuralnet/nneval.h"

class SelfplayManager {
 public:
  SelfplayManager(
    int maxDataQueueSize,
    Logger* logger,
    int64_t logGamesEvery,
    bool autoCleanupAllButLatestIfUnused
  );
  ~SelfplayManager();

  SelfplayManager(const SelfplayManager& other);
  SelfplayManager& operator=(const SelfplayManager& other);
  SelfplayManager(SelfplayManager&& other);
  SelfplayManager& operator=(SelfplayManager&& other);

  //All below functions are internally synchronized and thread-safe.

  //SelfplayManager takes responsibility for deleting the data writers and closing and deleting sgfOut.
  //loadModelNoDataWritingLoop is for the manual writing interface
  void loadModelAndStartDataWriting(
    NNEvaluator* nnEval,
    TrainingDataWriter* tdataWriter,
    std::ofstream* sgfOut
  );
  void loadModelNoDataWritingLoop(
    NNEvaluator* nnEval,
    TrainingDataWriter* tdataWriter,
    std::ofstream* sgfOut
  );

  //NN queries summed across all the models managed by this manager over all time.
  uint64_t getTotalNumRowsProcessed() const;

  //For all of the below, model names are simply from nnEval->getModelName().

  //Models that aren't cleaned up yet are in the order from earliest to latest
  std::vector<std::string> modelNames() const;
  std::string getLatestModelName() const;
  bool hasModel(const std::string& modelName) const;
  size_t numModels() const;

  //Returns NULL if acquire failed (such as if that model was scheduled to be cleaned up or already cleaned up,).
  //Must call release when done, and cease using the NNEvaluator after that.
  NNEvaluator* acquireModel(const std::string& modelName);
  NNEvaluator* acquireLatest();
  //Release a model either by name or by the nnEval object that was returned.
  void release(const std::string& modelName);
  void release(NNEvaluator* nnEval);

  //Write lease: hold this while enqueuing data AFTER releasing the eval.
  //This prevents full cleanup from erasing the model while data is still being enqueued.
  //Pointer-keyed overloads are stable across in-place model reloads that change modelName.
  void acquireWriteLease(const std::string& modelName);
  void releaseWriteLease(const std::string& modelName);
  void acquireWriteLease(NNEvaluator* nnEval);
  void releaseWriteLease(NNEvaluator* nnEval);

  //Attempt to reload the latest model's NNEvaluator in place with a new model file,
  //without creating a new NNEvaluator/ModelData. Returns true on success.
  //The existing ModelData's modelName is updated to newModelName; its TrainingDataWriter
  //and sgfOut stay bound to the existing ModelData (data continues to flow into the
  //same output stream).
  bool tryReloadLatestInPlace(
    const std::string& newModelFileName,
    const std::string& newModelName,
    const std::string& expectedSha256
  );

  //Clean up any currently-unused models if their last usage was older than this many seconds ago.
  void cleanupUnusedModelsOlderThan(double seconds);
  //Clear the evaluation caches of any models that are currently unused.
  void clearUnusedModelCaches();
  void noteModelEvalResourcesFreed(NNEvaluator* nnEval);
  int getModelAcquireCount(NNEvaluator* nnEval) const;
  void noteModelBackendResourcesFreed(NNEvaluator* nnEval);

  //====================================================================================
  //These should only be called by a thread that has currently acquired the model.

  //Increment a counter and maybe log some stats
  void countOneGameStarted(NNEvaluator* nnEval);

  //SelfplayManager takes responsibility for deleting the gameData once written.
  //Use these only if loadModelAndStartDataWriting was used to start the model.
  void enqueueDataToWrite(const std::string& modelName, FinishedGameData* gameData);
  void enqueueDataToWrite(NNEvaluator* nnEval, FinishedGameData* gameData);

  //Use these if loadModelNoDataWritingLoop was used to start the model.
  void withDataWriters(
    NNEvaluator* nnEval,
    const std::function<void(TrainingDataWriter* tdataWriter, std::ofstream* sgfOut)>& f
  );

  //Dump per-model acquire/writeRef/freed counters for debugging cleanup progression.
  void debugDumpModelStats();

  //====================================================================================

  //For internal use
  struct ModelData {
    std::string modelName;
    NNEvaluator* nnEval;
    int64_t gameStartedCount;
    double lastReleaseTime;
    bool hasDataWriteLoop;

    ThreadSafeQueue<FinishedGameData*> finishedGameQueue;
    int acquireCount;
    int writeRefCount;
    bool evalResourcesFreed;
    bool backendCleanupQueued;
    bool backendResourcesFreed;

    TrainingDataWriter* tdataWriter;
    std::ofstream* sgfOut;

    ModelData(
      const std::string& name, NNEvaluator* neval, int maxDataQueueSize,
      TrainingDataWriter* tdWriter, std::ofstream* sOut,
      double initialLastReleaseTime,
      bool hasDataWriteLoop
    );
    ~ModelData();
  };

 private:
  const int maxDataQueueSize;
  Logger* logger;
  const int64_t logGamesEvery;
  const bool autoCleanupAllButLatestIfUnused;

  const ClockTimer timer;

  mutable std::mutex managerMutex;
  std::vector<ModelData*> modelDatas;
  int numDataWriteLoopsActive;
  std::condition_variable dataWriteLoopsAreDone;

  uint64_t totalNumRowsProcessed;

  NNEvaluator* acquireModelAlreadyLocked(SelfplayManager::ModelData* foundData);
  void releaseAlreadyLocked(SelfplayManager::ModelData* foundData);
  void maybeAutoCleanupAlreadyLocked(std::vector<NNEvaluator*>& backendCleanupEvals);
  void finishPendingBackendCleanup(const std::vector<NNEvaluator*>& backendCleanupEvals);
  void runDataWriteLoopImpl(ModelData* modelData);

 public:
  //For internal use
  void runDataWriteLoop(ModelData* modelData);

};

#endif //PROGRAM_SELFPLAYMANAGER_H_
