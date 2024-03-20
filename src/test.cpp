#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "registry.hpp"
#include "test.hpp"

namespace test {

using namespace std;

#define EPS 1e-10

template <typename T> T *alloc(size_t n) {
  T *arr = (T *)aligned_alloc(n, n * sizeof(T));
  assert(arr != nullptr);
  return arr;
}

double *rand_vec(env_t &env, size_t n) {
  double *arr = alloc<double>(n);
  for (size_t i = 0; i < n; ++i) {
    arr[i] = env.dist(env.rd);
  }
  return arr;
};

data_t *random_data(env_t &env) {
  data_t *d = (data_t *)malloc(sizeof(data_t));
  assert(d != nullptr);

  d->x = rand_vec(env, env.n);
  d->y = rand_vec(env, env.n);
  d->d = rand_vec(env, env.n);
  d->n = env.n;

  return d;
}

void free_data(data_t *data) {
  free(data->x);
  free(data->y);
  free(data->d);
}

bool compare(registry::func_t a, registry::func_t b, env_t &env) {
  double d0[env.n], d1[env.n];
  data_t *data = random_data(env);
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

std::vector<env_t> envs;

void add_env(env_t env) { envs.push_back(env); }

void all() {
  auto baseline = registry::begin();
  auto iter = next(baseline);

  for (env_t env : envs) {
    // skip the baseline
    for (; iter != registry::end(); iter = next(iter)) {
      if (!compare(baseline->second, iter->second, env)) {
        cout << "ERR\t`" << iter->first << "` does not match the baseline"
             << endl;
        exit(1);
      }
    }
  }
  cout << "OK\tall implementations match the baseline" << endl;
}

} // namespace test
