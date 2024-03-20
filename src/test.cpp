#include <cassert>
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

double *rand_vec(env_t &e, size_t n) {
  double *arr = alloc<double>(n);
  for (size_t i = 0; i < n; ++i) {
    arr[i] = e.dist(e.rd);
  }
  return arr;
};

bool compare(registry::func_t a, registry::func_t b, env_t &env) {
  double *x = rand_vec(env, env.n);
  double *y = rand_vec(env, env.n);
  double *d0 = rand_vec(env, env.n), *d1 = rand_vec(env, env.n);
  a(x, y, d0, env.n);
  b(x, y, d1, env.n);

  for (size_t i = 0; i < env.n; ++i) {
    if (abs(d0[i] - d1[i]) > EPS) {
      free(x);
      free(y);
      free(d0);
      free(d1);
      return false;
    }
  }

  free(x);
  free(y);
  free(d0);
  free(d1);
  return true;
}

std::vector<env_t> envs;

void add_env(env_t env) { envs.push_back(env); }

void all() {
  std::vector<registry::entry_t *> &impls = registry::all();
  registry::entry_t *baseline = impls[0];

  for (env_t env : envs) {
    for (registry::entry_t *e : registry::all()) {
      if (e == baseline)
        continue;

      if (!compare(baseline->f, e->f, env)) {
        cout << "ERR\t`" << e->name << "` does not match the baseline" << endl;
        exit(1);
      }
    }
  }
  cout << "OK\tall implementations match the baseline" << endl;
}

} // namespace test
