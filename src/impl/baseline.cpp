#include "impl/baseline.hpp"

namespace baseline {

void mul(double *x, double *y, double *d, size_t n) {
  for (size_t i = 0; i < n; ++i)
    d[i] = x[i] * y[i];
}

} // namespace baseline
