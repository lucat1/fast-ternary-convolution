#ifndef _TEST_HPP
#define _TEST_HPP

#include <random>

#include "registry.hpp"

namespace test {

typedef struct test_env {
  // problem parameters
  size_t n;

  // random generators for the problem inputs
  std::uniform_real_distribution<double> dist;
  std::mt19937 rd;
} env_t;

typedef struct data {
  double *x;
  double *y;
  double *d;
  size_t n;
} data_t;

double *rand_vec(env_t &env, size_t n);

data_t *random_data(env_t &env);

void free_data(data_t *data);

bool compare(registry::func_t a, registry::func_t b);

void add_env(env_t env);

void all();

} // namespace test

#endif // _TEST_HPP
