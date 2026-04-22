#include "../core/global.h"
#include "../core/datetime.h"
#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../dataio/sgf.h"
#include "../dataio/trainingwrite.h"
#include "../dataio/loadmodel.h"
#include "../neuralnet/modelversion.h"
#include "../search/asyncbot.h"
#include "../program/setup.h"
#include "../program/play.h"
#include "../program/selfplaymanager.h"
#include "../command/commandline.h"
#include "../main.h"

#include <chrono>
#include <csignal>

using namespace std;

static std::atomic<bool> sigReceived(false);
static std::atomic<bool> shouldStop(false);
static void signalHandler(int signal)
{
  if(signal == SIGINT || signal == SIGTERM) {
    sigReceived.store(true);
    shouldStop.store(true);
  }
}

//-----------------------------------------------------------------------------------------


int MainCmds::selfplay(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string modelsDir;
  string outputDir;
  int64_t maxGamesTotal = ((int64_t)1) << 62;
  try {
    KataGoCommandLine cmd("Generate training data via self play.");
    cmd.addConfigFileArg("","");
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<string> modelsDirArg("","models-dir","Dir to poll and load models from",true,string(),"DIR");
    TCLAP::ValueArg<string> outputDirArg("","output-dir","Dir to output files",true,string(),"DIR");
    TCLAP::ValueArg<string> maxGamesTotalArg("","max-games-total","Terminate after this many games",false,string(),"NGAMES");
    cmd.add(modelsDirArg);
    cmd.add(outputDirArg);
    cmd.add(maxGamesTotalArg);
    cmd.parseArgs(args);

    modelsDir = modelsDirArg.getValue();
    outputDir = outputDirArg.getValue();
    string maxGamesTotalStr = maxGamesTotalArg.getValue();
    if(maxGamesTotalStr != "") {
      bool suc = Global::tryStringToInt64(maxGamesTotalStr,maxGamesTotal);
      if(!suc || maxGamesTotal <= 0)
        throw StringError("-max-games-total must be a positive integer");
    }

    auto checkDirNonEmpty = [](const char* flag, const string& s) {
      if(s.length() <= 0)
        throw StringError("Empty directory specified for " + string(flag));
    };
    checkDirNonEmpty("models-dir",modelsDir);
    checkDirNonEmpty("output-dir",outputDir);

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  MakeDir::make(outputDir);
  MakeDir::make(modelsDir);

  Logger logger(&cfg);
  //Log to random file name to better support starting/stopping as well as multiple parallel runs
  logger.addFile(outputDir + "/log" + DateTime::getCompactDateTimeString() + "-" + Global::uint64ToHexString(seedRand.nextUInt64()) + ".log");

  logger.write("Self Play Engine starting...");
  logger.write(string("Git revision: ") + Version::getGitRevision());

  //Load runner settings
  const int numGameThreads = cfg.getInt("numGameThreads",1,16384);
  const string gameSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  //Width and height of the board to use when writing data, typically 19
  const int dataBoardLen = cfg.getInt("dataBoardLen",3,Board::MAX_LEN);
  const int inputsVersion =
    cfg.contains("inputsVersion") ?
    cfg.getInt("inputsVersion",0,10000) :
    NNModelVersion::getInputsVersion(NNModelVersion::defaultModelVersion);
  //Max number of games that we will allow to be queued up and not written out
  const int maxDataQueueSize = cfg.getInt("maxDataQueueSize",1,1000000);
  const int maxRowsPerTrainFile = cfg.getInt("maxRowsPerTrainFile",1,100000000);
  const double firstFileRandMinProp = cfg.getDouble("firstFileRandMinProp",0.0,1.0);

  const int64_t logGamesEvery = cfg.getInt64("logGamesEvery",1,1000000);

  const bool switchNetsMidGame = cfg.getBool("switchNetsMidGame");
  const SearchParams baseParams = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_OTHER);
  WaitableFlag shouldPause;
  std::atomic<bool> shouldAbortGamesForReload(false);
#if defined(USE_TENSORRT_BACKEND)
  const char* refitEnv = std::getenv("KATAGO_TRT_REFIT");
  const bool trtRefitEnabled = (refitEnv != nullptr && string(refitEnv) == "1");
  // With TRT refit, model loading is fast (~183ms) and two evaluators can
  // safely coexist — no need to abort in-flight games.
  // Without refit, full TRT engine rebuild (~7 min) requires serialized reload.
  const bool useSerializedModelReload = !trtRefitEnabled;
#else
  const bool useSerializedModelReload = false;
#endif
  const bool allowMidgameNetSwitch = switchNetsMidGame && !useSerializedModelReload;
  if(useSerializedModelReload) {
    logger.write("TensorRT selfplay will serialize model reloads and may abort in-flight games during model updates");
    if(switchNetsMidGame)
      logger.write("TensorRT-safe reload disables midgame model switching");
  }
  else if(allowMidgameNetSwitch) {
    logger.write("TRT refit enabled: mid-game model switching active, no game abort on model update");
  }

  //Initialize object for randomizing game settings and running games
  const bool isDistributed = false;
  PlaySettings playSettings = PlaySettings::loadForSelfplay(cfg, isDistributed);
  GameRunner* gameRunner = new GameRunner(cfg, playSettings, logger);
  bool autoCleanupAllButLatestIfUnused = true;
  SelfplayManager* manager = new SelfplayManager(maxDataQueueSize, &logger, logGamesEvery, autoCleanupAllButLatestIfUnused);

  const int minBoardXSizeUsed = gameRunner->getGameInitializer()->getMinBoardXSize();
  const int minBoardYSizeUsed = gameRunner->getGameInitializer()->getMinBoardYSize();
  const int maxBoardXSizeUsed = gameRunner->getGameInitializer()->getMaxBoardXSize();
  const int maxBoardYSizeUsed = gameRunner->getGameInitializer()->getMaxBoardYSize();

  Setup::initializeSession(cfg);

  //Done loading!
  //------------------------------------------------------------------------------------
  logger.write("Loaded all config stuff, starting self play");
  if(!logger.isLoggingToStdout())
    cout << "Loaded all config stuff, starting self play" << endl;

  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);


  //Returns true if a new net was loaded.
  auto loadLatestNeuralNetIntoManager =
    [inputsVersion,&manager,maxRowsPerTrainFile,firstFileRandMinProp,dataBoardLen,
     &modelsDir,&outputDir,&logger,&cfg,numGameThreads,&shouldPause,&shouldAbortGamesForReload,
     switchNetsMidGame,useSerializedModelReload,allowMidgameNetSwitch,
     minBoardXSizeUsed,maxBoardXSizeUsed,minBoardYSizeUsed,maxBoardYSizeUsed](const string* lastNetName) -> bool {

    string modelName;
    string modelFile;
    string modelDir;
    time_t modelTime;
    bool foundModel = LoadModel::findLatestModel(modelsDir, logger, modelName, modelFile, modelDir, modelTime);

    //No new neural nets yet
    if(!foundModel || (lastNetName != NULL && *lastNetName == modelName))
      return false;
    if(modelName == "random" && lastNetName != NULL && *lastNetName != "random") {
      logger.write("WARNING: " + *lastNetName + " was the previous model, but now no model was found. Continuing with prev model instead of using random");
      return false;
    }

    logger.write("Found new neural net " + modelName);

    const int expectedConcurrentEvals = cfg.getInt("numSearchThreads") * numGameThreads;
    const bool defaultRequireExactNNLen = minBoardXSizeUsed == maxBoardXSizeUsed && minBoardYSizeUsed == maxBoardYSizeUsed;
    const int defaultMaxBatchSize = -1;
    const bool disableFP16 = false;
    const string expectedSha256 = "";

    Rand rand;
    NNEvaluator* nnEval = NULL;
    TrainingDataWriter* tdataWriter = NULL;
    ofstream* sgfOut = NULL;
    bool pausedForSafeHandoff = false;
    bool serializedReloadInProgress = false;

    // ========== IN-PLACE REFIT FAST PATH ==========
    // If TRT refit is enabled and we already have a live evaluator, skip creating a
    // second NNEvaluator/ModelData entirely. Reload the weights in place on the
    // existing evaluator, keep its tdataWriter/sgfOut bound to the existing ModelData.
    // This eliminates the double-evaluator state that causes GPU context and NNCache
    // to stack up across model switches — the root cause of the observed 322 MiB/switch
    // GPU leak and linear CPU RSS growth in the pure/overlap baselines.
    if(allowMidgameNetSwitch && lastNetName != NULL && manager->numModels() > 0) {
      bool reloaded = false;
      try {
        reloaded = manager->tryReloadLatestInPlace(modelFile, modelName, expectedSha256);
      } catch(const std::exception& e) {
        logger.write(string("In-place reload threw: ") + e.what() + "; falling back to full evaluator creation");
        reloaded = false;
      } catch(...) {
        logger.write("In-place reload threw unknown exception; falling back to full evaluator creation");
        reloaded = false;
      }
      if(reloaded) {
        logger.write("In-place reloaded latest evaluator to " + modelName);
        return true;
      }
      logger.write("In-place reload failed for " + modelName + "; falling back to full evaluator creation");
    }
    // ========== END FAST PATH ==========

    try {
      nnEval = Setup::initializeNNEvaluator(
        modelName,modelFile,expectedSha256,cfg,logger,rand,expectedConcurrentEvals,
        maxBoardXSizeUsed,maxBoardYSizeUsed,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
        Setup::SETUP_FOR_OTHER,
        false
      );

      if(useSerializedModelReload && lastNetName != NULL && manager->numModels() > 0) {
        serializedReloadInProgress = true;
        logger.write("Draining in-flight selfplay for TRT-safe model handoff to " + modelName);
        shouldAbortGamesForReload.store(true, std::memory_order_release);

        NNEvaluator* oldLatest = manager->acquireLatest();
        if(oldLatest != NULL) {
          logger.write("Draining previous latest model before enabling new TRT backend: " + oldLatest->getModelName());
          oldLatest->releaseHeavyResources();
          manager->noteModelEvalResourcesFreed(oldLatest);

          //Keep our own acquire lease alive throughout the drain. If we
          //release first and wait for acquireCount==0, maybeAutoCleanup can
          //queue backend teardown + Phase-3 erase, and the data-write loop
          //may delete the ModelData (and its NNEvaluator) while we still hold
          //oldLatest — a use-after-free on the later releaseRetiredBackendResources
          //call. Drain down to acquireCount==1 (our own lease) instead, then
          //free backend, then do a single final release that triggers
          //Phase-3 cleanup cleanly.
          int lastLoggedAcquireCount = -1;
          while(!shouldStop.load()) {
            int acquireCount = manager->getModelAcquireCount(oldLatest);
            if(acquireCount <= 1)
              break;
            if(acquireCount != lastLoggedAcquireCount) {
              logger.write(
                "Waiting for " + Global::intToString(acquireCount - 1) +
                " in-flight selfplay games to release retired model " + oldLatest->getModelName()
              );
              lastLoggedAcquireCount = acquireCount;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
          }
          if(!shouldStop.load()) {
            oldLatest->releaseRetiredBackendResources();
            manager->noteModelBackendResourcesFreed(oldLatest);
          }
          //Final release. After this, oldLatest may be freed by the data-write
          //loop at any time; do not touch it again.
          manager->release(oldLatest);
          oldLatest = NULL;
        }
      }
      else if(switchNetsMidGame && !allowMidgameNetSwitch && lastNetName != NULL && manager->numModels() > 0) {
        pausedForSafeHandoff = true;
        logger.write("Pausing selfplay for TRT-safe model handoff to " + modelName);
        shouldPause.set(true);

        NNEvaluator* oldLatest = manager->acquireLatest();
        if(oldLatest != NULL) {
          logger.write("Draining previous latest model before enabling new TRT backend: " + oldLatest->getModelName());
          oldLatest->releaseHeavyResources();
          oldLatest->releaseRetiredBackendResources();
          manager->release(oldLatest);
        }
      }
      // When allowMidgameNetSwitch is true (TRT refit mode), skip both serialized
      // and paused handoff. Game threads will switch to the new model naturally at
      // move boundaries via checkForNewNNEval. The old model stays fully functional
      // until acquireCount reaches 0, then maybeAutoCleanup handles resource release.

      nnEval->spawnServerThreads();
      logger.write("Loaded latest neural net " + modelName + " from: " + modelFile);

      string modelOutputDir = outputDir + "/" + modelName;
      string sgfOutputDir = modelOutputDir + "/sgfs";
      string tdataOutputDir = modelOutputDir + "/tdata";

      //Try repeatedly to make directories, in case the filesystem is unhappy with us as we try to make the same dirs as another process.
      //Wait a random amount of time in between each failure.
      int maxTries = 5;
      for(int i = 0; i<maxTries; i++) {
        bool success = false;
        try {
          MakeDir::make(modelOutputDir);
          MakeDir::make(sgfOutputDir);
          MakeDir::make(tdataOutputDir);
          success = true;
        }
        catch(const StringError& e) {
          logger.write(string("WARNING, error making directories, trying again shortly: ") + e.what());
          success = false;
        }

        if(success)
          break;
        else {
          if(i == maxTries-1) {
            logger.write("ERROR: Could not make selfplay model directories, is something wrong with the filesystem?");
            //Just give up and wait for the next model.
            if(serializedReloadInProgress || pausedForSafeHandoff) {
              shouldAbortGamesForReload.store(false, std::memory_order_release);
              if(pausedForSafeHandoff)
                shouldPause.set(false);
              logger.write("Resuming selfplay after failed TRT-safe handoff");
            }
            delete nnEval;
            return false;
          }
          double sleepTime = 10.0 + rand.nextDouble() * 30.0;
          std::this_thread::sleep_for(std::chrono::duration<double>(sleepTime));
          continue;
        }
      }

      {
        ofstream out;
        FileUtils::open(out,modelOutputDir + "/" + "selfplay-" + Global::uint64ToHexString(rand.nextUInt64()) + ".cfg");
        out << cfg.getContents();
        out.close();
      }

      //Note that this inputsVersion passed here is NOT necessarily the same as the one used in the neural net self play, it
      //simply controls the input feature version for the written data
      tdataWriter = new TrainingDataWriter(
        tdataOutputDir, inputsVersion, maxRowsPerTrainFile, firstFileRandMinProp, dataBoardLen, dataBoardLen, Global::uint64ToHexString(rand.nextUInt64()));
      if(sgfOutputDir.length() > 0) {
        sgfOut = new ofstream();
        FileUtils::open(*sgfOut, sgfOutputDir + "/" + Global::uint64ToHexString(rand.nextUInt64()) + ".sgfs");
      }

      logger.write("Model loading loop thread loaded new neural net " + nnEval->getModelName());
      manager->loadModelAndStartDataWriting(nnEval, tdataWriter, sgfOut);
      nnEval = NULL;
      tdataWriter = NULL;
      sgfOut = NULL;
      if(serializedReloadInProgress || pausedForSafeHandoff) {
        shouldAbortGamesForReload.store(false, std::memory_order_release);
        if(pausedForSafeHandoff)
          shouldPause.set(false);
        logger.write("Resuming selfplay after TRT-safe model handoff to " + modelName);
      }
      return true;
    }
    catch(...) {
      shouldAbortGamesForReload.store(false, std::memory_order_release);
      if(pausedForSafeHandoff && shouldPause.get())
        shouldPause.set(false);
      if(nnEval != NULL)
        delete nnEval;
      if(tdataWriter != NULL)
        delete tdataWriter;
      if(sgfOut != NULL)
        delete sgfOut;
      throw;
    }
  };

  //Initialize the initial neural net
  {
    bool success = loadLatestNeuralNetIntoManager(NULL);
    if(!success)
      throw StringError("Either could not load latest neural net or access/write appopriate directories");
  }

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  //Shared across all game loop threads
  std::atomic<int64_t> numGamesStarted(0);
  ForkData* forkData = new ForkData();
  auto gameLoop = [
    &gameRunner,
    &manager,
    &logger,
    &shouldPause,
    &shouldAbortGamesForReload,
    allowMidgameNetSwitch,
    &numGamesStarted,
    &forkData,
    maxGamesTotal,
    &baseParams,
    &gameSeedBase
  ](int threadIdx) {
    WaitableFlag* shouldPausePtr = &shouldPause;

    string prevModelName;
    Rand thisLoopSeedRand;
    while(true) {
      if(shouldStop.load())
        break;
      if(shouldPausePtr->get())
        shouldPausePtr->waitUntilFalse();
      if(shouldStop.load())
        break;

      NNEvaluator* nnEval = manager->acquireLatest();
      if(nnEval == NULL) {
        if(shouldPausePtr->get() || shouldAbortGamesForReload.load(std::memory_order_acquire)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          continue;
        }
        throw StringError("Selfplay game loop could not acquire latest model");
      }
      if(shouldPausePtr->get() || shouldAbortGamesForReload.load(std::memory_order_acquire)) {
        manager->release(nnEval);
        continue;
      }

      if(prevModelName != nnEval->getModelName()) {
        prevModelName = nnEval->getModelName();
        logger.write("Game loop thread " + Global::intToString(threadIdx) + " starting game on new neural net: " + prevModelName);
      }

      //Callback that runGame will call periodically to ask us if we have a new neural net
      std::function<NNEvaluator*()> checkForNewNNEval = [&manager,&nnEval,&prevModelName,&logger,&threadIdx]() -> NNEvaluator* {
        NNEvaluator* newNNEval = manager->acquireLatest();
        assert(newNNEval != NULL);
        if(newNNEval == nnEval) {
          manager->release(newNNEval);
          return NULL;
        }
        manager->release(nnEval);

        nnEval = newNNEval;
        prevModelName = nnEval->getModelName();
        logger.write("Game loop thread " + Global::intToString(threadIdx) + " changing midgame to new neural net: " + prevModelName);
        return nnEval;
      };

      FinishedGameData* gameData = NULL;
      int64_t gameIdx = numGamesStarted.fetch_add(1,std::memory_order_acq_rel);
      if(gameIdx < maxGamesTotal) {
        bool abortedForReloadThisGame = false;
        auto shouldStopFunc = [&shouldAbortGamesForReload,&abortedForReloadThisGame]() noexcept {
          if(shouldStop.load())
            return true;
          if(shouldAbortGamesForReload.load(std::memory_order_acquire)) {
            abortedForReloadThisGame = true;
            return true;
          }
          return false;
        };

        manager->countOneGameStarted(nnEval);
        MatchPairer::BotSpec botSpecB;
        botSpecB.botIdx = 0;
        botSpecB.botName = nnEval->getModelName();
        botSpecB.nnEval = nnEval;
        botSpecB.baseParams = baseParams;
        MatchPairer::BotSpec botSpecW = botSpecB;

        string seed = gameSeedBase + ":" + Global::uint64ToHexString(thisLoopSeedRand.nextUInt64());
        gameData = gameRunner->runGame(
          seed, botSpecB, botSpecW, forkData, NULL, logger,
          shouldStopFunc,
          shouldPausePtr,
          (allowMidgameNetSwitch ? checkForNewNNEval : nullptr),
          nullptr,
          nullptr
        );

        if(gameData == NULL && abortedForReloadThisGame && !shouldStop.load()) {
          manager->release(nnEval);
          continue;
        }
      }

      //NULL gamedata will happen when the game is interrupted by shouldStop, which means we should also stop.
      //Or when we run out of total games.
      bool shouldContinue = gameData != NULL;
      // Use pointer-keyed lease/enqueue: an in-place reload may rename the ModelData's
      // modelName between game start and enqueue, but the NNEvaluator* is stable.
      if(gameData != NULL)
        manager->acquireWriteLease(nnEval);
      NNEvaluator* nnEvalForWrite = nnEval;
      manager->release(nnEval);
      if(gameData != NULL) {
        try {
          manager->enqueueDataToWrite(nnEvalForWrite,gameData);
        }
        catch(...) {
          manager->releaseWriteLease(nnEvalForWrite);
          throw;
        }
        manager->releaseWriteLease(nnEvalForWrite);
      }

      if(!shouldContinue)
        break;
    }

    logger.write("Game loop thread " + Global::intToString(threadIdx) + " terminating");
  };
  auto gameLoopProtected = [&logger,&gameLoop](int threadIdx) {
    Logger::logThreadUncaught("game loop", &logger, [&](){ gameLoop(threadIdx); });
  };

  //Looping thread for polling for new neural nets and loading them in
  std::mutex modelLoadMutex;
  std::condition_variable modelLoadSleepVar;
  auto modelLoadLoop = [&modelLoadMutex,&modelLoadSleepVar,&logger,&manager,&loadLatestNeuralNetIntoManager]() {
    logger.write("Model loading loop thread starting");

    while(true) {
      if(shouldStop.load())
        break;
      string lastNetName = manager->getLatestModelName();
      bool success = loadLatestNeuralNetIntoManager(&lastNetName);
      (void)success;

      if(shouldStop.load())
        break;

      //Sleep for a while and then re-poll
      std::unique_lock<std::mutex> lock(modelLoadMutex);
      modelLoadSleepVar.wait_for(lock, std::chrono::seconds(20), [](){return shouldStop.load();});
    }

    logger.write("Model loading loop thread terminating");
  };
  auto modelLoadLoopProtected = [&logger,&modelLoadLoop]() {
    Logger::logThreadUncaught("model load loop", &logger, modelLoadLoop);
  };

  vector<std::thread> threads;
  threads.reserve(numGameThreads);
  for(int i = 0; i<numGameThreads; i++) {
    threads.emplace_back(gameLoopProtected,i);
  }
  std::thread modelLoadLoopThread(modelLoadLoopProtected);

  //DEBUG: optional monitor thread, gated by env flag. Dumps per-model
  //acquireCount / writeRefCount / freed-flags every 10s. Off by default;
  //set KATAGO_DEBUG_MODEL_STATS=1 to enable during cleanup-path debugging.
  const char* debugStatsEnv = std::getenv("KATAGO_DEBUG_MODEL_STATS");
  const bool debugStatsEnabled = (debugStatsEnv != nullptr && string(debugStatsEnv) == "1");
  std::thread debugMonitorThread;
  if(debugStatsEnabled) {
    debugMonitorThread = std::thread([&manager]() {
      while(!shouldStop.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        if(shouldStop.load()) break;
        manager->debugDumpModelStats();
      }
    });
  }

  //Wait for all game threads to stop
  for(int i = 0; i<threads.size(); i++)
    threads[i].join();

  //If by now somehow shouldStop is not true, set it to be true since all game threads are toast
  shouldStop.store(true);

  //Wake up the model loading thread rather than waiting for it to wake up on its own, and
  //wait for it to die.
  {
    //Lock so that we don't race where we notify the loading thread to wake when it's still in
    //its own critical section but not yet slept, and to ensure the two agree on shouldStop.
    std::lock_guard<std::mutex> lock(modelLoadMutex);
    modelLoadSleepVar.notify_all();
  }
  modelLoadLoopThread.join();
  if(debugMonitorThread.joinable())
    debugMonitorThread.join();

  //At this point, nothing else except possibly data write loops are running, within the selfplay manager.
  delete manager;

  //Delete and clean up everything else
  NeuralNet::globalCleanup();
  delete forkData;
  delete gameRunner;
  ScoreValue::freeTables();

  if(sigReceived.load())
    logger.write("Exited cleanly after signal");
  logger.write("All cleaned up, quitting");
  return 0;
}
