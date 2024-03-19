#include <iostream>

#include "baseline.hpp"
#include "vectorized.hpp"

#define N 1024

int main(int argc, char *argv[]) {
  double x[N], y[N], d[N];
  baseline::mul(x, y, d, N);
  vectorized::mul(x, y, d, N);
  return 0;
}
