#include "asl/fib.hh"
#include "benchmark/benchmark.h"

static void BM_Fib(benchmark::State &state) {
  while (state.KeepRunning()) {
    asl::fib(state.range(0));
  }
}
BENCHMARK(BM_Fib)->Args({20})->Args({21})->Args({22})->Args({23})->Args({24});

BENCHMARK_MAIN();
