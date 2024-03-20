#include <iostream>
#include <random>

#include "impl/baseline.hpp"
#include "impl/vectorized.hpp"

#include "bench.hpp"
#include "registry.hpp"
#include "test.hpp"

static const std::map<registry::name_t, registry::func_t> functions = {
    {"baseline", baseline::mul}, {"vectorized", vectorized::mul}};

int main(int argc, char *argv[]) {
  std::random_device rd;
  std::mt19937 gen(rd());

  registry::define(functions);
  test::env_t env1 =
      test::env_t{.n = 1024 * 256,
                  .dist = std::uniform_real_distribution<double>(0, 1),
                  .rd = gen};
  test::add_env(env1);
  test::all();
  std::cout << bench::measure("baseline", baseline::mul, env1) << std::endl;
  std::cout << bench::measure("vectorized", vectorized::mul, env1) << std::endl;

  return 0;
}
