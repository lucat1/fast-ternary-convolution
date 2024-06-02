#pragma once

#define tensor7d_get_123_4567(data, dim4, dim5, dim6, dim7, i, j)	\
  ((data)[(i) * (dim4) * (dim5) * (dim6) * (dim7) + (j)])

#define tensor7d_set(data, dim2, dim3, dim4, dim5, dim6, dim7, value, i, j, k, l, m, n, o) \
  ((data)[(((((((i) * (dim2) + (j)) * (dim3) + (k)) * (dim4) + (l)) * (dim5) + (m)) * (dim6) + (n)) * (dim7) + (o))] = (value))

#define tensor5d_get(data, dim2, dim3, dim4, dim5, i, j, k, l, m) \
  ((data)[(((((i) * (dim2) + (j)) * (dim3) + (k)) * (dim4) + (l)) * (dim5) + (m))])

#define tensor5d_set(data, dim2, dim3, dim4, dim5, value, i, j, k, l, m) \
  ((data)[(((((i) * (dim2) + (j)) * (dim3) + (k)) * (dim4) + (l)) * (dim5) + (m))] = (value))

#define tensor5d_get_1_2345(data, dim2, dim3, dim4, dim5, i, j)	\
  ((data)[(i) * (dim2) * (dim3) * (dim4) * (dim5) + (j)])

#define tensor4d_get(data, dim2, dim3, dim4, i, j, k, l)	\
  ((data)[(((i) * (dim2) + (j)) * (dim3) + (k)) * (dim4) + (l)])

#define tensor4d_set(data, dim2, dim3, dim4, value, i, j, k, l)		\
  ((data)[(((i) * (dim2) + (j)) * (dim3) + (k)) * (dim4) + (l)] = (value))

// TODO this is inconsistent, value should follow after data and dim4 (same in tmacro1)
#define tensor4d_set_123_4(value, data, dim4, i, j)	\
  ((data)[(i) * (dim4) + (j)] = (value))

#define tensor1d_get(data, i) \
  ((data)[(i)])
