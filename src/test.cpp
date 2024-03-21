#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "registry.hpp"
#include "test.hpp"

namespace test {

#define EPS (1e-3)

bool compare(registry::func_t a, registry::func_t b, registry::env_t &env) {
  double d0[env.n], d1[env.n];
  registry::data_t *data = registry::random_data(env);
  memcpy(d0, data->d, data->n);
  memcpy(d1, data->d, data->n);

  a(data->x, data->y, d0, env.n);
  b(data->x, data->y, d1, env.n);

  for (size_t i = 0; i < env.n; ++i) {
    if (abs(d0[i] - d1[i]) > EPS) {
      free_data(data);
      return false;
    }
  }

  free_data(data);
  return true;
}

void all() {
  using namespace registry;

  auto baseline = functions::begin();
  auto func = next(baseline);

  for (auto env = environments::begin(); env != environments::end();
       env = next(env)) {
    // skip the baseline
    for (; func != functions::end(); func = next(func)) {
      if (!compare(baseline->second, func->second, env->second)) {
        std::cout << "ERR\t`" << func->first << "` does not match the baseline"
                  << std::endl;
        exit(1);
      }
    }
  }
  std::cout << "OK\tall implementations match the baseline" << std::endl;
}

} // namespace test
