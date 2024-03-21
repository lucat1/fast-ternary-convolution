#include <iostream>
#include <random>

#include "impl/baseline.hpp"
#include "impl/vectorized.hpp"

#include "bench.hpp"
#include "registry.hpp"
#include "test.hpp"

std::random_device rd;
std::mt19937 gen(rd());

static const std::map<registry::name_t, registry::func_t> functions = {
    {registry::BASELINE_NAME, baseline::mul}, {"vectorized", vectorized::mul}};
static const std::map<registry::name_t, registry::env_t> environments = {
    {"xs", registry::env_t{.n = 1024 * 256,
                           .dist = std::uniform_real_distribution<double>(0, 1),
                           .rd = gen}},
    {"sm", registry::env_t{.n = 1024 * 1024,
                           .dist = std::uniform_real_distribution<double>(0, 1),
                           .rd = gen}}};

int main(int argc, char *argv[]) {

  registry::functions::set(functions);
  registry::environments::set(environments);

  test::all();
  bench::all();

  return 0;
}
