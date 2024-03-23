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
  size_t output_size = registry::output_size(env);
  std::cout << output_size << std::endl;

  float *output0 = registry::rand_real_vec(env, output_size);
  float *output1 = registry::rand_real_vec(env, output_size);
  registry::data_t *data = registry::random_data(env);

  memcpy(output0, data->output, output_size);
  memcpy(output1, data->output, output_size);

  a(env.type, data->btn_cnt1, data->input, data->input_height,
    data->input_width, data->padding_height, data->padding_width,
    data->quant_threshold, data->num_channels, data->quant_weights,
    data->batch_size, data->stride_height, data->stride_width,
    data->kernel_number, data->kernel_height, data->kernel_width,
    data->relu_alpha, data->output);
  b(env.type, data->btn_cnt1, data->input, data->input_height,
    data->input_width, data->padding_height, data->padding_width,
    data->quant_threshold, data->num_channels, data->quant_weights,
    data->batch_size, data->stride_height, data->stride_width,
    data->kernel_number, data->kernel_height, data->kernel_width,
    data->relu_alpha, data->output);

  for (size_t i = 0; i < output_size; ++i) {
    if (abs(output0[i] - output1[i]) > EPS) {
      free_data(data);
      return false;
    }
  }

  free_data(data);
  return true;
}

void all() {
  using namespace registry;

  auto baseline = functions::get(BASELINE_NAME);

  for (auto env = environments::begin(); env != environments::end();
       env = next(env)) {
    // skip the baseline
    for (auto func = functions::begin(); func != functions::end();
         func = next(func)) {
      if (func->second == baseline)
        continue;
      if (!compare(baseline, func->second, env->second)) {
        std::cout << "ERR\t`" << func->first << "` does not match the baseline"
                  << std::endl;
        exit(1);
      }
    }
  }
  std::cout << "TEST\tall ok" << std::endl;
}

} // namespace test
