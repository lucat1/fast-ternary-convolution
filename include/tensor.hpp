#pragma once
#include "alloc.hpp"

// Toggle inlining of get and set methods
// NOTE: this is already enabled when you compile with `make` or `make optimize`
// #define INLINE

// TODO Remove unused getters/setters

// NOTE It may be a good idea to simplify the index computations when we inline
// them.
//   Ideally we compare them using Compiler Explorer.

// Implements a five dimensional tensor for basic types.
// T (probably) must have a copy constructor for get()
template <typename T> class Tensor5D {
public:
  // cannot be const as move constructor may set it to nullptr
  T *data;
  const size_t dim1;
  const size_t dim2;
  const size_t dim3;
  const size_t dim4;
  const size_t dim5;

  // Construct a new five dimensional tensor.
  // Input:
  //  dim{i}: the size of the i-th dimension
  //  zero: if true, initializes the memory with zero.
  Tensor5D(const size_t dim1, const size_t dim2, const size_t dim3,
           const size_t dim4, const size_t dim5, const bool zero)
      : data(zero ? alloc::calloc<T>(dim1 * dim2 * dim3 * dim4 * dim5)
                  : alloc::alloc<T>(dim1 * dim2 * dim3 * dim4 * dim5)),
        dim1(dim1), dim2(dim2), dim3(dim3), dim4(dim4), dim5(dim5) {}

  // Destructor: automatically cleans up memory when the object
  // leaves the scope.
  ~Tensor5D() {
    // pointer may be zero due to the move constructor moving
    // the data out
    if (data != nullptr) {
      alloc::free(data);
    }
  }

#ifdef INLINE
  inline
#endif
      T *
      addr(const size_t i, const size_t j, const size_t k, const size_t l,
           const size_t m) const {
    assert(i < dim1);
    assert(j < dim2);
    assert(k < dim3);
    assert(l < dim4);
    assert(m < dim5);

    return &data[(i * (dim2 * dim3 * dim4 * dim5)) +
                 (j * (dim3 * dim4 * dim5)) + (k * (dim4 * dim5)) +
                 (l * (dim5)) + m];
  }

#ifdef INLINE
  inline
#endif
      T
      get(const size_t i, const size_t j, const size_t k, const size_t l,
          const size_t m) const {
    assert(i < dim1);
    assert(j < dim2);
    assert(k < dim3);
    assert(l < dim4);
    assert(m < dim5);

    return data[(i * (dim2 * dim3 * dim4 * dim5)) + (j * (dim3 * dim4 * dim5)) +
                (k * (dim4 * dim5)) + (l * (dim5)) + m];
  }

  // Get element from the tensor interpreting it as a 2D tensor
  // where the last 4 dimensions are merged.
#ifdef INLINE
  inline
#endif
      T
      get_1_2345(const size_t i, const size_t j) const {
    assert(i < dim1);
    assert(j < dim2 * dim3 * dim4 * dim5);

    return data[i * dim2 * dim3 * dim4 * dim5 + j];
  }

#ifdef INLINE
  inline
#endif
      T
      get_123_4_5(const size_t i, const size_t j, const size_t k) const {
    assert(i < dim1 * dim2 * dim3);
    assert(j < dim4);
    assert(k < dim5);
    return data[i * dim4 * dim5 + j * dim5 + k];
  }

#ifdef INLINE
  inline
#endif
      void
      set(const T value, const size_t i, const size_t j, const size_t k,
          const size_t l, const size_t m) {
    assert(i < dim1);
    assert(j < dim2);
    assert(k < dim3);
    assert(l < dim4);
    assert(m < dim5);

    data[(i * (dim2 * dim3 * dim4 * dim5)) + (j * (dim3 * dim4 * dim5)) +
         (k * (dim4 * dim5)) + (l * (dim5)) + m] = value;
  }

  // Rule of Five: either define all of the essential
  // operations, or none.
  // - default constructor: unnecessary
  // - do not allow copying (expensive; do it manually if
  // needed)
  // - no move assignment: want dimensions to stay const if
  // possible
  // - move constructor: not sure whether it will be used, but
  // at least the
  //   compiler needs it (cannot guarantee NRVO)

  // Default constructor
  Tensor5D() = delete;
  // Copy constructor
  Tensor5D(const Tensor5D &) = delete;
  // Copy assignment
  Tensor5D &operator=(const Tensor5D &) = delete;
  // Move constructor
  Tensor5D(Tensor5D &&other)
      : data(other.data), dim1(other.dim1), dim2(other.dim2), dim3(other.dim3),
        dim4(other.dim4), dim5(other.dim5) {
    // if we do not change this to nullptr, destructing this and
    // other will (probably) lead to a double free.
    other.data = nullptr;
  }
  // Move assignment
  Tensor5D &operator=(Tensor5D &&) = delete;
};

template <typename T> class Tensor3D {
public:
  // cannot be const as move constructor may set it to nullptr
  T *data;
  const size_t dim1;
  const size_t dim2;
  const size_t dim3;

  // Construct a new five dimensional tensor.
  // Input:
  //  dim{i}: the size of the i-th dimension
  //  zero: if true, initializes the memory with zero.
  Tensor3D(const size_t dim1, const size_t dim2, const size_t dim3,
           const bool zero)
      : data(zero ? alloc::calloc<T>(dim1 * dim2 * dim3)
                  : alloc::alloc<T>(dim1 * dim2 * dim3)),
        dim1(dim1), dim2(dim2), dim3(dim3) {}

  // Destructor: automatically cleans up memory when the object
  // leaves the scope.
  ~Tensor3D() {
    // pointer may be zero due to the move constructor moving
    // the data out
    if (data != nullptr) {
      alloc::free(data);
    }
  }

#ifdef INLINE
  inline
#endif
      T *
      addr(const size_t i, const size_t j, const size_t k) const {
    assert(i < dim1);
    assert(j < dim2);
    assert(k < dim3);

    return &data[(i * (dim2 * dim3)) + (j * (dim3)) + k];
  }

#ifdef INLINE
  inline
#endif
      T
      get(const size_t i, const size_t j, const size_t k) const {
    assert(i < dim1);
    assert(j < dim2);
    assert(k < dim3);

    return data[(i * (dim2 * dim3)) + (j * (dim3)) + k];
  }

  // Get an element from the tensor by interpreting it as a 1D
  // tensor where dimensions 1, 2 and 3 are merged
#ifdef INLINE
  inline
#endif
      T
      get_123(const size_t i) const {
    assert(i < dim1 * dim2 * dim3);
    return data[i];
  }

#ifdef INLINE
  inline
#endif
      void
      set(const T value, const size_t i, const size_t j, const size_t k) {
    assert(i < dim1);
    assert(j < dim2);
    assert(k < dim3);

    data[(i * (dim2 * dim3)) + (j * (dim3)) + k] = value;
  }

  // Rule of Five: either define all of the essential
  // operations, or none.
  // - default constructor: unnecessary
  // - do not allow copying (expensive; do it manually if
  // needed)
  // - no move assignment: want dimensions to stay const if
  // possible
  // - move constructor: not sure whether it will be used, but
  // at least the
  //   compiler needs it (cannot guarantee NRVO)

  // Default constructor
  Tensor3D() = delete;
  // Copy constructor
  Tensor3D(const Tensor3D &) = delete;
  // Copy assignment
  Tensor3D &operator=(const Tensor3D &) = delete;
  // Move constructor
  Tensor3D(Tensor3D &&other)
      : data(other.data), dim1(other.dim1), dim2(other.dim2), dim3(other.dim3) {
    // if we do not change this to nullptr, destructing this and
    // other will (probably) lead to a double free.
    other.data = nullptr;
  }
  // Move assignment
  Tensor3D &operator=(Tensor3D &&) = delete;
};

// Implements a four dimensional tensor for basic types.
template <typename T> class Tensor4D {
public:
  T *data;
  const size_t dim1;
  const size_t dim2;
  const size_t dim3;
  const size_t dim4;

  // Construct a new four dimensional tensor.
  // Input:
  //  dim{i}: the size of the i-th dimension
  //  zero: if true, initializes the memory with zero.
  Tensor4D(const size_t dim1, const size_t dim2, const size_t dim3,
           const size_t dim4, const bool zero)
      : data(zero ? alloc::calloc<T>(dim1 * dim2 * dim3 * dim4)
                  : alloc::alloc<T>(dim1 * dim2 * dim3 * dim4)),
        dim1(dim1), dim2(dim2), dim3(dim3), dim4(dim4) {}

#ifdef INLINE
  inline
#endif
      T
      get(const size_t i, const size_t j, const size_t k,
          const size_t l) const {
    assert(i < dim1);
    assert(j < dim2);
    assert(k < dim3);
    assert(l < dim4);

    return data[(i * (dim2 * dim3 * dim4)) + (j * (dim3 * dim4)) + (k * dim4) +
                l];
  }

#ifdef INLINE
  inline
#endif
      void
      set(const T value, const size_t i, const size_t j, const size_t k,
          const size_t l) {
    assert(i < dim1);
    assert(j < dim2);
    assert(k < dim3);
    assert(l < dim4);

    data[(i * (dim2 * dim3 * dim4)) + (j * (dim3 * dim4)) + (k * dim4) + l] =
        value;
  }

  // Set an element in the tensor by interpreting it as a 2D
  // tensor where the first 3 dimensions have been merged.
#ifdef INLINE
  inline
#endif
      void
      set_123_4(const T value, const size_t i, const size_t j) {
    assert(i < dim1 * dim2 * dim3);
    assert(j < dim4);

    data[i * dim4 + j] = value;
  }

  // Destructor
  ~Tensor4D() {
    if (data != nullptr) {
      alloc::free(data);
    }
  }

  // Default constructor
  Tensor4D() = delete;
  // Copy constructor
  Tensor4D(const Tensor4D &) = delete;
  // Copy assignment
  Tensor4D &operator=(const Tensor4D &) = delete;
  // Move constructor
  Tensor4D(Tensor4D &&other)
      : data(other.data), dim1(other.dim1), dim2(other.dim2), dim3(other.dim3),
        dim4(other.dim4) {
    other.data = nullptr;
  }
  // Move assignment
  Tensor4D &operator=(Tensor4D &&) = delete;
};

// Reshapes a tensor from (N, H, W, C) to (N, C, H, W)
template <typename T> Tensor4D<T> reshape_nhwc_nchw(const Tensor4D<T> &src) {
  const size_t N = src.dim1;
  const size_t H = src.dim2;
  const size_t W = src.dim3;
  const size_t C = src.dim4;

  Tensor4D<float> dest = Tensor4D<float>(N, C, H, W, false);

  for (size_t in = 0; in < N; in++) {
    for (size_t ih = 0; ih < H; ih++) {
      for (size_t iw = 0; iw < W; iw++) {
        for (size_t ic = 0; ic < C; ic++) {
          T val = src.get(in, ih, iw, ic);
          dest.set(val, in, ic, ih, iw);
        }
      }
    }
  }

  return dest;
}

// Implements a two dimensional tensor for basic types.
template <typename T> class Tensor2D {
public:
  T *data;
  const size_t dim1;
  const size_t dim2;

  // Construct a new two dimensional tensor.
  // Input:
  //  dim{i}: the size of the i-th dimension
  //  zero: if true, initializes the memory with zero.
  Tensor2D(const size_t dim1, const size_t dim2, const bool zero)
      : data(zero ? alloc::calloc<T>(dim1 * dim2)
                  : alloc::alloc<T>(dim1 * dim2)),
        dim1(dim1), dim2(dim2) {}

#ifdef INLINE
  inline
#endif
      T
      get(const size_t i, const size_t j) const {
    assert(i < dim1);
    assert(j < dim2);

    return data[i * dim2 + j];
  }

#ifdef INLINE
  inline
#endif
      void
      set(const T value, const size_t i, const size_t j) {
    assert(i < dim1);
    assert(j < dim2);

    data[i * dim2 + j] = value;
  }

  // Destructor
  ~Tensor2D() {
    if (data != nullptr) {
      alloc::free(data);
    }
  }

  // Default constructor
  Tensor2D() = delete;
  // Copy constructor
  Tensor2D(const Tensor2D &) = delete;
  // Copy assignment
  Tensor2D &operator=(const Tensor2D &) = delete;
  // Move constructor
  Tensor2D(Tensor2D &&other)
      : data(other.data), dim1(other.dim1), dim2(other.dim2) {
    other.data = nullptr;
  }
  // Move assignment
  Tensor2D &operator=(Tensor2D &) = delete;
};

// Implements a one dimensional tensor for basic types.
template <typename T> class Tensor1D {
public:
  T *data;
  const size_t size;

  // Construct a new one dimensional tensor.
  // Input:
  //  size: the size of the tensor
  //  zero: if true, initializes the memory with zero.
  Tensor1D(const size_t size, const bool zero)
      : data(zero ? alloc::calloc<T>(size) : alloc::alloc<T>(size)),
        size(size) {}

#ifdef INLINE
  inline
#endif
      T
      get(const size_t i) const {
    assert(i < size);
    return data[i];
  }

  // Destructor
  ~Tensor1D() {
    if (data != nullptr) {
      alloc::free(data);
    }
  }

  // Default constructor
  Tensor1D() = delete;
  // Copy constructor
  Tensor1D(const Tensor1D &) = delete;
  // Copy assignment
  Tensor1D &operator=(const Tensor1D &) = delete;
  // Move constructor
  Tensor1D(Tensor1D &&other) : data(other.data), size(other.size) {
    other.data = nullptr;
  }
  // Move assignment
  Tensor1D &operator=(Tensor1D &&) = delete;
};

// Implements a seven dimensional tensor for basic types.
// T (probably) must have a copy constructor for get()
template <typename T> class Tensor7D {
public:
  // cannot be const as move constructor may set it to nullptr
  T *data;
  const size_t dim1;
  const size_t dim2;
  const size_t dim3;
  const size_t dim4;
  const size_t dim5;
  const size_t dim6;
  const size_t dim7;

  // Construct a new seven dimensional tensor.
  // Input:
  //  dim{i}: the size of the i-th dimension
  //  zero: if true, initializes the memory with zero.
  Tensor7D(const size_t dim1, const size_t dim2, const size_t dim3,
           const size_t dim4, const size_t dim5, const size_t dim6,
           const size_t dim7, const bool zero)
      : data(zero ? alloc::calloc<T>(dim1 * dim2 * dim3 * dim4 * dim5 * dim6 *
                                     dim7)
                  : alloc::alloc<T>(dim1 * dim2 * dim3 * dim4 * dim5 * dim6 *
                                    dim7)),
        dim1(dim1), dim2(dim2), dim3(dim3), dim4(dim4), dim5(dim5), dim6(dim6),
        dim7(dim7) {}

  // Destructor: automatically cleans up memory when the object
  // leaves the scope.
  ~Tensor7D() {
    if (data != nullptr) {
      alloc::free(data);
    }
  }

#ifdef INLINE
  inline
#endif
      T
      get(const size_t i, const size_t j, const size_t k, const size_t l,
          const size_t m, const size_t n, const size_t o) const {
    assert(i < dim1);
    assert(j < dim2);
    assert(k < dim3);
    assert(l < dim4);
    assert(m < dim5);
    assert(n < dim6);
    assert(o < dim7);

    return data[(i * (dim2 * dim3 * dim4 * dim5 * dim6 * dim7)) +
                (j * (dim3 * dim4 * dim5 * dim6 * dim7)) +
                (k * (dim4 * dim5 * dim6 * dim7)) + (l * (dim5 * dim6 * dim7)) +
                (m * (dim6 * dim7)) + (n * dim7) + o];
  }

  // Get an element from the tensor by interpreting it as a 2D
  // tensor where dimensions 1, 2 and 3 are merged, and 4, 5, 6
  // and 7 are merged.
#ifdef INLINE
  inline
#endif
      T
      get_123_4567(const size_t i, const size_t j) const {
    assert(i < dim1 * dim2 * dim3);
    assert(j < dim4 * dim5 * dim6 * dim7);
    return data[i * dim4 * dim5 * dim6 * dim7 + j];
  }

#ifdef INLINE
  inline
#endif
      void
      set(const T value, const size_t i, const size_t j, const size_t k,
          const size_t l, const size_t m, const size_t n, const size_t o) {
    assert(i < dim1);
    assert(j < dim2);
    assert(k < dim3);
    assert(l < dim4);
    assert(m < dim5);
    assert(n < dim6);
    assert(o < dim7);

    data[(i * (dim2 * dim3 * dim4 * dim5 * dim6 * dim7)) +
         (j * (dim3 * dim4 * dim5 * dim6 * dim7)) +
         (k * (dim4 * dim5 * dim6 * dim7)) + (l * (dim5 * dim6 * dim7)) +
         (m * (dim6 * dim7)) + (n * dim7) + o] = value;
  }

  // Default constructor
  Tensor7D() = delete;
  // Copy constructor
  Tensor7D(const Tensor7D &) = delete;
  // Copy assignment
  Tensor7D &operator=(const Tensor7D &) = delete;
  // Move constructor
  Tensor7D(Tensor7D &&other)
      : data(other.data), dim1(other.dim1), dim2(other.dim2), dim3(other.dim3),
        dim4(other.dim4), dim5(other.dim5), dim6(other.dim6), dim7(other.dim7) {
    other.data = nullptr;
  }
  // Move assignment
  Tensor7D &operator=(Tensor7D &&) = delete;
};
