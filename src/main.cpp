#include <iostream>
#include <random>

#include "impl/baseline.hpp"
#include "impl/vectorized.hpp"

#include "registry.hpp"
#include "test.hpp"

int main(int argc, char *argv[]) {
  std::random_device rd;
  std::mt19937 gen(rd());

  registry::def("baseline", baseline::mul);
  registry::def("vectorized", vectorized::mul);

  test::add_env(
      test::env_t{.n = 1024,
                  .dist = std::uniform_real_distribution<double>(0, 1),
                  .rd = gen});
  test::all();

  return 0;
}
