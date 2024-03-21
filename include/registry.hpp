#ifndef _REGISTRY_HPP
#define _REGISTRY_HPP

#include <cstddef>
#include <map>
#include <random>
#include <utility>

namespace registry {

typedef void (*func_t)(double const *, double const *, double *, size_t);
typedef const char *name_t;
typedef struct env {
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

namespace functions {

void set(std::map<name_t, func_t> entries);
std::map<name_t, func_t>::iterator begin();
std::map<name_t, func_t>::iterator end();

} // namespace functions

namespace environments {

void set(std::map<name_t, env_t> entries);
std::map<name_t, env_t>::iterator begin();
std::map<name_t, env_t>::iterator end();

} // namespace environments

} // namespace registry

#endif // _REGISTRY_HPP
