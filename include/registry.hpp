#ifndef _REGISTRY_HPP
#define _REGISTRY_HPP

#include "common.hpp"

#include <cassert>
#include <cstddef>
#include <cstring> // memset
#include <exception>
#include <map>
#include <random>
#include <string>
#include <utility>

namespace registry {

typedef enum conv_type {
  TNN = 0,
  TBN = 1,
  BTN = 2,
  BNN = 3,
  CONV_TYPES = 4
} conv_type_t;

typedef void (*func_t)(registry::conv_type_t, int *, float *, uint32_t,
                       uint32_t, uint32_t, uint32_t, float *, int, int64_t *,
                       uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                       uint32_t, float, float *);
typedef const std::string name_t;

typedef struct env {
  // problem parameters
  uint32_t input_height;
  uint32_t input_width;
  uint32_t batch_size;

  conv_type_t type;
  uint32_t num_channels;
  uint32_t kernel_number;
  uint32_t kernel_height;
  uint32_t kernel_width;
  uint32_t padding_size;
  uint32_t stride_size;

  // random generators for the problem inputs
  std::uniform_real_distribution<float> real_dist;
  std::uniform_int_distribution<int64_t> int_dist;
  std::mt19937 rd;
} env_t;

size_t input_size(env_t &env);
size_t output_size(env_t &env);

typedef struct data {
  // network type
  conv_type_t type;
  int *btn_cnt1;

  float *input;
  uint32_t input_height;
  uint32_t input_width;
  uint32_t padding_height;
  uint32_t padding_width;

  uint32_t num_channels;
  float *quant_threshold;
  int64_t *quant_weights;

  uint32_t batch_size;
  uint32_t stride_height;
  uint32_t stride_width;
  uint32_t kernel_number;
  uint32_t kernel_height;
  uint32_t kernel_width;

  float relu_alpha;
  float *output;
} data_t;

constexpr size_t ALIGNMENT = 32;

size_t __make32(size_t needed_size, size_t alignment);

template <typename T> T *alloc(size_t n) {
  T *arr = static_cast<T *>(
      aligned_alloc(ALIGNMENT, __make32(n * sizeof(T), ALIGNMENT)));
  assert(arr != nullptr);
  return arr;
}

template <typename T> T *calloc(size_t n) {
  T *arr = static_cast<T *>(
      aligned_alloc(ALIGNMENT, __make32(n * sizeof(T), ALIGNMENT)));
  assert(arr != nullptr);
  memset(arr, 0, n * sizeof(T));
  return static_cast<T *>(arr);
}

float *rand_real_vec(env_t &env, size_t n);
int64_t *rand_int_vec(env_t &env, size_t n);
float *const_vec(size_t n, float val);

data_t *random_data(env_t &env);

void free_data(data_t *data);

constexpr std::string BASELINE_NAME = "baseline";

namespace functions {

void set(std::map<name_t, func_t> entries);
func_t get(name_t name);
std::map<name_t, func_t>::iterator begin();
std::map<name_t, func_t>::iterator end();

} // namespace functions

namespace environments {

void set(std::map<name_t, env_t> entries);
env_t get(name_t name);
std::map<name_t, env_t>::iterator begin();
std::map<name_t, env_t>::iterator end();

} // namespace environments

} // namespace registry

#endif // _REGISTRY_HPP
