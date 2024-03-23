#include "bench.hpp"
#include "registry.hpp"
#include "table.hpp"
#include "tsc.hpp"

#include <cstdio>
#include <iostream>
#include <list>
#include <map>

#define NR 32
#define CYCLES_REQUIRED 1e8
#define REP 30

namespace bench {

double measure(registry::func_t f, registry::env_t &env) {
  double cycles = 0.;
  size_t num_runs = 100;
  double multiplier = 1;
  uint64_t start, end;

  registry::data_t *data = registry::random_data(env);

  // Warm-up phase: we determine a number of executions that allows
  // the code to be executed for at least CYCLES_REQUIRED cycles.
  // This helps excluding timing overhead when measuring small runtimes.
  do {
    num_runs = num_runs * multiplier;
    start = start_tsc();
    for (size_t i = 0; i < num_runs; i++) {
      f(data->x, data->y, data->d, data->n);
    }
    end = stop_tsc(start);

    cycles = (double)end;
    multiplier = (CYCLES_REQUIRED) / (cycles);

  } while (multiplier > 2);

  std::list<double> cycles_list;

  // Actual performance measurements repeated REP times.
  // We simply store all results and compute medians during post-processing.
  double total_cycles = 0;
  for (size_t j = 0; j < REP; j++) {

    start = start_tsc();
    for (size_t i = 0; i < num_runs; ++i) {
      f(data->x, data->y, data->d, data->n);
    }
    end = stop_tsc(start);

    cycles = ((double)end) / num_runs;
    total_cycles += cycles;

    cycles_list.push_back(cycles);
  }
  total_cycles /= REP;

  cycles = total_cycles;
  return cycles;
}

void all() {
  using namespace registry;

  table::hsep();
  table::htitle("Benchmarks");
  table::hsep();
  table::row({"impl", "cycles", "env", "speedup"}, {});

  for (auto env = environments::begin(); env != environments::end();
       env = next(env)) {
    auto baseline = functions::get(BASELINE_NAME);
    uint64_t baseline_cycles = measure(baseline, env->second);

    table::row({BASELINE_NAME, std::to_string(baseline_cycles).c_str(),
                env->first, "1x"},
               {});

    for (auto func = functions::begin(); func != functions::end();
         func = next(func)) {
      // skip the baseline
      if (func->second == baseline)
        continue;
      uint64_t cycles = measure(func->second, env->second);
      float speedup = (float)baseline_cycles / (float)cycles;
      char speedup_str[5];
      sprintf(speedup_str, "%.2f", speedup);
      table::row({func->first, std::to_string(cycles).c_str(), env->first,
                  speedup_str},
                 {});
    }
    table::hsep();
  }
}

} // namespace bench
