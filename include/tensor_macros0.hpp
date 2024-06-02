#pragma once
#include "alloc.hpp"

// performance macros
#define tensor1d_get(t, i) (t.data[i])
#define tensor4d_get(t, i, j, k, l)                                            \
  (t.data[((i) * (t.dim2 * t.dim3 * t.dim4)) + ((j) * (t.dim3 * t.dim4)) +     \
          ((k) * t.dim4) + (l)])
#define tensor7d_get(t, i, j, k, l, m, n, o)                                   \
  (t.data[((i) * (t.dim2 * t.dim3 * t.dim4 * t.dim5 * t.dim6 * t.dim7)) +      \
          ((j) * (t.dim3 * t.dim4 * t.dim5 * t.dim6 * t.dim7)) +               \
          ((k) * (t.dim4 * t.dim5 * t.dim6 * t.dim7)) +                        \
          ((l) * (t.dim5 * t.dim6 * t.dim7)) + ((m) * (t.dim6 * t.dim7)) +     \
          ((n) * t.dim7) + (o)])
#define tensor7d_addr(t, i, j, k, l, m, n, o)                                  \
  (&t.data[((i) * (t.dim2 * t.dim3 * t.dim4 * t.dim5 * t.dim6 * t.dim7)) +     \
           ((j) * (t.dim3 * t.dim4 * t.dim5 * t.dim6 * t.dim7)) +              \
           ((k) * (t.dim4 * t.dim5 * t.dim6 * t.dim7)) +                       \
           ((l) * (t.dim5 * t.dim6 * t.dim7)) + ((m) * (t.dim6 * t.dim7)) +    \
           ((n) * t.dim7) + (o)])

#define tensor4d_set(t, v, i, j, k, l)                                         \
  (t.data[((i) * (t.dim2 * t.dim3 * t.dim4)) + ((j) * (t.dim3 * t.dim4)) +     \
          ((k) * t.dim4) + (l)] = (v))
#define tensor7d_set(t, v, i, j, k, l, m, n, o)                                \
  (t.data[((i) * (t.dim2 * t.dim3 * t.dim4 * t.dim5 * t.dim6 * t.dim7)) +      \
          ((j) * (t.dim3 * t.dim4 * t.dim5 * t.dim6 * t.dim7)) +               \
          ((k) * (t.dim4 * t.dim5 * t.dim6 * t.dim7)) +                        \
          ((l) * (t.dim5 * t.dim6 * t.dim7)) + ((m) * (t.dim6 * t.dim7)) +     \
          ((n) * t.dim7) + (o)] = (v))
