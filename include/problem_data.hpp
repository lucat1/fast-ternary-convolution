#pragma once
#include "alloc.hpp"
#include "problem_parameters.hpp"

// We don't want to confuse this with std::vector
template <typename T> class Matrix1D {
public:
  const size_t size;
  T *const data;

  Matrix1D(size_t size, bool zero)
      : size(size),
        data(zero ? alloc::calloc<T>(size) : alloc::alloc<T>(size)) {}
  ~Matrix1D() {
    if (data != nullptr)
      alloc::free(data);
  }
};

template <typename T> class Matrix2D {
public:
  const Shape2D shape;
  T *const data;

  Matrix2D(size_t height, size_t width, bool zero)
      : shape(height, width), data(zero ? alloc::calloc<T>(shape.size)
                                        : alloc::alloc<T>(shape.size)) {}
  Matrix2D(Shape2D shape, bool zero)
      : Matrix2D(shape.height, shape.width, zero) {}
  ~Matrix2D() {
    if (data != nullptr)
      alloc::free(data);
  }
};

class Shape4D {
public:
  const size_t fst_dim;
  const size_t snd_dim;
  const size_t height;
  const size_t width;

  // precomputed fst_dim * snd_dim * height * width
  const size_t size;

  Shape4D(size_t fst_dim, size_t snd_dim, size_t height, size_t width);
};

template <typename T> class Matrix4D {
public:
  const Shape4D shape;
  T *const data;

  Matrix4D(size_t fst_dim, size_t snd_dim, size_t height, size_t width,
           bool zero)
      : shape(fst_dim, snd_dim, height, width),
        data(zero ? alloc::calloc<T>(shape.size)
                  : alloc::alloc<T>(shape.size)) {}
  Matrix4D(Shape4D shape, bool zero)
      : Matrix4D(shape.fst_dim, shape.snd_dim, shape.height, shape.width,
                 zero) {}
  ~Matrix4D() {
    if (data != nullptr)
      alloc::free(data);
  }
};

class Shape5D {
public:
  const size_t fst_dim;
  const size_t snd_dim;
  const size_t trd_dim;
  const size_t height;
  const size_t width;

  // precomputed fst_dim * snd_dim * trd_dim * height * width
  const size_t size;

  Shape5D(size_t fst_dim, size_t snd_dim, size_t trd_dim, size_t height,
          size_t width);
};

template <typename T> class Matrix5D {
public:
  const Shape5D shape;
  T *const data;

  Matrix5D(size_t fst_dim, size_t snd_dim, size_t trd_dim, size_t height,
           size_t width, bool zero)
      : shape(fst_dim, snd_dim, trd_dim, height, width),
        data(zero ? alloc::calloc<T>(shape.size)
                  : alloc::alloc<T>(shape.size)) {}
  Matrix5D(Shape5D shape, bool zero)
      : Matrix5D(shape.fst_dim, shape.snd_dim, shape.trd_dim, shape.height,
                 shape.width, zero) {}
  ~Matrix5D() {
    if (data != nullptr)
      alloc::free(data);
  }
};

class Data : public Parameters {
public:
  // shapes after quantization + packing
  const Shape2D packed_input_size;
  const Shape2D packed_kernel_size;
  const size_t packed_channels;

  const Matrix4D<float> x; // input data
  const Matrix1D<float> quant_threshold;
  const Matrix5D<int64_t> quant_weights;
  const Matrix1D<int> btn_cnt;
  const Matrix4D<float> y; // output data

  Data(Parameters params);
};
