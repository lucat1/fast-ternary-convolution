#include "bench.hpp"
#include "test.hpp"
#include "tsc.hpp"

#include <list>

#define NR 32
#define CYCLES_REQUIRED 1e8
#define REP 30
#define EPS (1e-3)

namespace bench {

double measure(registry::name_t name, registry::func_t f, test::env_t env) {
  double cycles = 0.;
  size_t num_runs = 100;
  double multiplier = 1;
  uint64_t start, end;

  test::data_t *data = test::random_data(env);

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

} // namespace bench
