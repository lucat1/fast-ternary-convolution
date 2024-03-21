#include "registry.hpp"

#include <cassert>
#include <cstdlib>
#include <map>

namespace registry {

template <typename T> T *alloc(size_t n) {
  T *arr = static_cast<T *>(aligned_alloc(32, n * sizeof(T)));
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

namespace functions {

std::map<name_t, func_t> entries;

void set(std::map<name_t, func_t> map) { entries = map; }

std::map<name_t, func_t>::iterator begin() { return begin(entries); }
std::map<name_t, func_t>::iterator end() { return end(entries); }

} // namespace functions

namespace environments {

std::map<name_t, env_t> entries;

void set(std::map<name_t, env_t> map) { entries = map; }

std::map<name_t, env_t>::iterator begin() { return begin(entries); }
std::map<name_t, env_t>::iterator end() { return end(entries); }

} // namespace environments

} // namespace registry
