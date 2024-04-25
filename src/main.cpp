#include <cstdio>
#include <iostream>
#include <limits>
#include <random>

#include "impl/baseline/tab.hpp"

#include "bench.hpp"
#include "registry.hpp"
#include "test.hpp"
#include "verify.hpp"

std::random_device rd;
std::mt19937 gen(rd());

auto real_dist = std::uniform_real_distribution<float>(0, 1);
auto int_dist = std::uniform_int_distribution<int64_t>(
    std::numeric_limits<int>::min(), std::numeric_limits<int>::max());

static const std::map<registry::name_t, registry::func_t> functions = {
    {registry::BASELINE_NAME, baseline::conv}};

static const std::map<registry::name_t, registry::env_t> environments = {};

typedef struct test_case {
  uint32_t c;
  uint32_t h;
  uint32_t w;
  uint32_t kn;
  uint32_t kh;
  uint32_t kw;
  uint32_t p;
  uint32_t s;
} test_case_t;

struct test_case cases[] = {
    {64, 56, 56, 64, 3, 3, 1, 1},    {64, 56, 56, 128, 3, 3, 1, 1},
    {128, 28, 28, 128, 3, 3, 1, 1},  {128, 28, 28, 256, 3, 3, 1, 1},
    {256, 14, 14, 256, 3, 3, 1, 1},  {256, 14, 14, 512, 3, 3, 1, 1},

    {80, 224, 224, 80, 3, 3, 1, 1},  {80, 224, 224, 80, 3, 3, 1, 2},
    {80, 224, 224, 80, 3, 3, 1, 3},  {80, 224, 224, 80, 3, 3, 1, 4},

    {512, 56, 56, 256, 1, 1, 0, 1},  {512, 56, 56, 256, 3, 3, 1, 1},
    {512, 56, 56, 256, 5, 5, 2, 1},  {512, 56, 56, 256, 7, 7, 3, 1},
    {512, 56, 56, 256, 9, 9, 3, 1},  {512, 56, 56, 256, 11, 11, 3, 1},

    {2000, 1, 1, 4000, 1, 1, 0, 1},  {4000, 1, 1, 8000, 1, 1, 0, 1},
    {8000, 1, 1, 16000, 1, 1, 0, 1}, {16000, 1, 1, 32000, 1, 1, 0, 1},
};

int main(void) {
  std::map<registry::name_t, registry::env_t> environments;
  for (uint32_t batch_size = 1; batch_size < 4; ++batch_size) {
    for (size_t icase = 0; icase < sizeof(cases) / sizeof(struct test_case);
         icase++) {
      test_case_t c = cases[icase];

      char *name = (char *)malloc(128 * sizeof(char));
      assert(name != nullptr);
      snprintf(name, 128, "%d-%d (%dx%d)", batch_size, c.c, c.w, c.h);

      // TODO Is this a fair assumption?
      assert(c.w == c.h);
      environments.insert(
          {name, registry::env_t{
                     .input_height = c.h,
                     .input_width = c.w, // common image size
                     .batch_size = batch_size,
                     .type = registry::TNN,
                     .num_channels = c.c,
                     .kernel_number = c.kn,
                     .kernel_height = c.kh,
                     .kernel_width = c.kw,
                     .padding_size = c.p,
                     .stride_size = c.s,

                     .real_dist = std::uniform_real_distribution<float>(0, 1),
                     .int_dist = std::uniform_int_distribution<int64_t>(
                         std::numeric_limits<int>::min(),
                         std::numeric_limits<int>::max()),
                     .rd = gen}});
    }
  }

  registry::functions::set(functions);
  registry::environments::set(environments);

  test::all();
  verify::verify();
  bench::all();

  return 0;
}
