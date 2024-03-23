#ifndef _BASELINE_TAB_ACTIVATION_HPP
#define _BASELINE_TAB_ACTIVATION_HPP

#include <vector>

namespace baseline {

template <typename T>
std::vector<float> PReLU(T *x, int n, int c, int h, int w, float alpha) {
  std::vector<float> y = std::vector<float>(n * c * h * w);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < c; j++) {
      for (int k = 0; k < h; k++) {
        for (int a = 0; a < w; a++) {
          T current = x[((i * c + j) * h + k) * w + a];
          if (current > 0)
            y[((i * c + j) * h + k) * w + a] = current;
          else
            y[((i * c + j) * h + k) * w + a] = current * alpha;
        }
      }
    }
  }

  return y;
}

} // namespace baseline

#endif // _BASELINE_TAB_ACTIVATION_HPP
