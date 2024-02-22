set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY    https://github.com/google/benchmark.git
  GIT_TAG           344117638c8ff7e239044fd0fa7085839fc03021 # version 1.8.3
)

FetchContent_MakeAvailable(
  googlebenchmark
)

set(bench_cc
  fib_bench.cc
)

add_executable(bench_asl ${bench_cc})

target_link_libraries(bench_asl
  benchmark
  ${ASL_LIB_NAME}
)
