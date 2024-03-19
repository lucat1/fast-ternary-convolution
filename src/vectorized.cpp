#include <cstddef>
#include <immintrin.h>

namespace vectorized {

void mul(double *x, double *y, double *d, size_t n) {
  __m256d xi, yi, di;

  for (size_t i = 0; i < n; i += 4) {
    xi = _mm256_load_pd(x + i);
    yi = _mm256_load_pd(y + i);

    di = _mm256_mul_pd(xi, yi);

    _mm256_store_pd(d + i, di);
  }
}

} // namespace vectorized
