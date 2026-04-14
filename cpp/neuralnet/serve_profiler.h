#pragma once
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>

// Runtime toggle: set KATAGO_SERVE_PROFILE=1 to enable
inline std::atomic<bool> g_serveProfileEnabled{false};

// GPU overlap callback: executed during async GPU computation in backends
// that support it (e.g., TRT enqueueV3). Set by serve() to overlap notify
// with GPU work. Cleared after execution.
inline thread_local std::function<void()> g_gpuOverlapCallback;

inline void initServeProfiler() {
  const char* env = std::getenv("KATAGO_SERVE_PROFILE");
  if(env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0)
    g_serveProfileEnabled.store(true, std::memory_order_relaxed);
}

// Thread-local sub-phase timings filled by getOutput(), read by serve()
struct GetOutputTimings {
  double input_asm_us;
  double h2d_us;
  double gpu_compute_us;
  double d2h_us;
  double output_post_us;
};
inline thread_local GetOutputTimings g_goTimings = {};

struct ServeProfileStats {
  using Clock = std::chrono::steady_clock;
  using TP = Clock::time_point;
  static double us(TP a, TP b) { return std::chrono::duration<double,std::micro>(b-a).count(); }

  double queue_pop_us = 0;
  double alloc_us = 0;
  double sym_us = 0;
  double get_output_us = 0;
  double notify_us = 0;
  double bufupd_us = 0;
  // sub-phases (copied from g_goTimings each batch)
  double input_asm_us = 0;
  double h2d_us = 0;
  double gpu_us = 0;
  double d2h_us = 0;
  double outpost_us = 0;

  int64_t rows = 0, batches = 0;
  int minbs = 999999, maxbs = 0;
  TP t0;

  static constexpr int INTERVAL = 500;

  void addBatch(int bs) {
    if(batches == 0) t0 = Clock::now();
    rows += bs; batches++;
    if(bs < minbs) minbs = bs;
    if(bs > maxbs) maxbs = bs;
    input_asm_us += g_goTimings.input_asm_us;
    h2d_us       += g_goTimings.h2d_us;
    gpu_us       += g_goTimings.gpu_compute_us;
    d2h_us       += g_goTimings.d2h_us;
    outpost_us   += g_goTimings.output_post_us;
  }

  bool due() const { return batches > 0 && batches % INTERVAL == 0; }

  void report(int gpu) {
    double wall_s = us(t0, Clock::now()) / 1e6;
    double N = (double)batches;
    double cycle = queue_pop_us + alloc_us + sym_us + get_output_us + notify_us + bufupd_us;
    auto ms  = [&](double v){ return v/N/1000.0; };
    auto pct = [&](double v){ return cycle>0 ? v/cycle*100.0 : 0.0; };
    fprintf(stderr,
      "\n[PROF GPU%d] %lld batches in %.1fs  %.1f batch/s  avg_bs=%.1f [%d,%d]\n"
      "  queue_pop    %7.3fms  %5.1f%%\n"
      "  alloc        %7.3fms  %5.1f%%\n"
      "  symmetry     %7.3fms  %5.1f%%\n"
      "  getOutput    %7.3fms  %5.1f%%\n"
      "    input_asm  %7.3fms  %5.1f%%\n"
      "    h2d        %7.3fms  %5.1f%%\n"
      "    gpu_comp   %7.3fms  %5.1f%%\n"
      "    d2h        %7.3fms  %5.1f%%\n"
      "    out_post   %7.3fms  %5.1f%%\n"
      "  notify       %7.3fms  %5.1f%%\n"
      "  buf_update   %7.3fms  %5.1f%%\n"
      "  CYCLE_TOTAL  %7.3fms\n",
      gpu, (long long)batches, wall_s, N/wall_s, (double)rows/N, minbs, maxbs,
      ms(queue_pop_us),  pct(queue_pop_us),
      ms(alloc_us),      pct(alloc_us),
      ms(sym_us),        pct(sym_us),
      ms(get_output_us), pct(get_output_us),
      ms(input_asm_us),  pct(input_asm_us),
      ms(h2d_us),        pct(h2d_us),
      ms(gpu_us),        pct(gpu_us),
      ms(d2h_us),        pct(d2h_us),
      ms(outpost_us),    pct(outpost_us),
      ms(notify_us),     pct(notify_us),
      ms(bufupd_us),     pct(bufupd_us),
      cycle/N/1000.0);
    fflush(stderr);
    *this = ServeProfileStats{};
  }
};
