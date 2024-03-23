#include "registry.hpp"

#include <cstdlib>
#include <map>

namespace registry {

float *rand_real_vec(env_t &env, size_t n) {
  float *arr = alloc<float>(n);
  for (size_t i = 0; i < n; ++i) {
    arr[i] = env.real_dist(env.rd);
  }
  return arr;
};

int64_t *rand_int_vec(env_t &env, size_t n) {
  int64_t *arr = alloc<int64_t>(n);
  for (size_t i = 0; i < n; ++i) {
    arr[i] = env.int_dist(env.rd);
  }
  return arr;
};

float *const_vec(env_t &env, size_t n, float val) {
  float *arr = alloc<float>(n);
  for (size_t i = 0; i < n; ++i) {
    arr[i] = val;
  }
  return arr;
};

size_t input_size(env_t &env) {
  return env.batch_size * env.num_channels * env.input_size * env.input_size;
}

size_t output_size(env_t &env) {
  uint32_t packed_height = env.input_size + 2 * env.padding_size;
  uint32_t packed_width = env.input_size + 2 * env.padding_size;
  uint32_t output_height =
      (packed_height - env.kernel_height + 1) / env.stride_size;
  uint32_t output_width =
      (packed_width - env.kernel_width + 1) / env.stride_size;

  return env.batch_size * env.num_channels * output_width * output_height;
}

data_t *random_data(env_t &env) {
  data_t *d = (data_t *)malloc(sizeof(data_t));
  assert(d != nullptr);

  d->type = env.type;
  d->btn_cnt1 = nullptr; // TODO: when we implement binary convolutions

  d->input = rand_real_vec(env, input_size(env));
  d->input_width = env.input_size;
  d->input_height = env.input_size;
  d->padding_height = env.padding_size;
  d->padding_width = env.padding_size;

  d->num_channels = env.num_channels;
  // TODO: why 1024? should be parameterized?
  d->quant_threshold = const_vec(env, 1024, 0.5);
  d->quant_weights = rand_int_vec(env, env.kernel_number * env.num_channels *
                                           env.kernel_width *
                                           env.kernel_height * BITS / CNTBITS);

  d->batch_size = env.batch_size;
  d->stride_height = env.stride_size;
  d->stride_width = env.stride_size;
  d->kernel_number = env.kernel_number;
  d->kernel_height = env.kernel_height;
  d->kernel_width = env.kernel_width;

  d->output = rand_real_vec(env, output_size(env));
  return d;
}

void free_data(data_t *data) {
  free(data->input);
  free(data->quant_threshold);
  free(data->quant_weights);
  free(data->output);
}

namespace functions {

std::map<name_t, func_t> entries;

void set(std::map<name_t, func_t> map) { entries = map; }
func_t get(name_t name) { return entries[name]; }
std::map<name_t, func_t>::iterator begin() { return begin(entries); }
std::map<name_t, func_t>::iterator end() { return end(entries); }

} // namespace functions

namespace environments {

std::map<name_t, env_t> entries;

void set(std::map<name_t, env_t> map) { entries = map; }
env_t get(name_t name) { return entries[name]; }
std::map<name_t, env_t>::iterator begin() { return begin(entries); }
std::map<name_t, env_t>::iterator end() { return end(entries); }

} // namespace environments

} // namespace registry
