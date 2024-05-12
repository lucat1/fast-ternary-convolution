#pragma once
#include "alloc.hpp"

// TODO Remove unused getters/setters

// NOTE It may be a good idea to simplify the index computations when we inline them.
//   Ideally we compare them using Compiler Explorer.

// Implements a five dimensional tensor for basic types.
// T (probably) must have a copy constructor for get()
template <typename T>
class Tensor5D {
public:
  // cannot be const as move constructor may set it to nullptr
  T* data;
  const size_t dim1;
  const size_t dim2;
  const size_t dim3;
  const size_t dim4;
  const size_t dim5;

  // Construct a new five dimensional tensor.
  // Input:
  //  dim{i}: the size of the i-th dimension
  //  zero: if true, initializes the memory with zero.
  Tensor5D(const size_t dim1, const size_t dim2, const size_t dim3, const size_t dim4,
	   const size_t dim5, const bool zero)
    : data(zero ? alloc::calloc<T>(dim1 * dim2 * dim3 * dim4 * dim5)
	   : alloc::alloc<T>(dim1 * dim2 * dim3 * dim4 * dim5)),
      dim1(dim1), dim2(dim2), dim3(dim3), dim4(dim4), dim5(dim5) {}

  // Destructor: automatically cleans up memory when the object leaves the scope.
  ~Tensor5D() {
    // pointer may be zero due to the move constructor moving the data out
    if (data != nullptr) {
      alloc::free(data);
    }
  }

  T get(const size_t i, const size_t j, const size_t k, const size_t l,
	   const size_t m) const {
    assert(i < dim1);
    assert(j < dim2);
    assert(k < dim3);
    assert(l < dim4);
    assert(m < dim5);

    return data[(i * (dim2 * dim3 * dim4 * dim5)) +
		(j * (dim3 * dim4 * dim5)) + (k * (dim4 * dim5)) + (l * (dim5)) + m];
  }

  void set(const T value, const size_t i, const size_t j, const size_t k, const size_t l,
	   const size_t m) {
    assert(i < dim1);
    assert(j < dim2);
    assert(k < dim3);
    assert(l < dim4);
    assert(m < dim5);

    data[(i * (dim2 * dim3 * dim4 * dim5)) +
	 (j * (dim3 * dim4 * dim5)) + (k * (dim4 * dim5)) + (l * (dim5)) + m] = value;
  }

  // Rule of Five: either define all of the essential operations, or none.
  // - default constructor: unnecessary
  // - do not allow copying (expensive; do it manually if needed)
  // - no move assignment: want dimensions to stay const if possible
  // - move constructor: not sure whether it will be used, but at least the
  //   compiler needs it (cannot guarantee NRVO)

  // Default constructor
  Tensor5D() =delete;
  // Copy constructor
  Tensor5D(const Tensor5D&) =delete;
  // Copy assignment
  Tensor5D& operator=(const Tensor5D&) =delete;
  // Move constructor
  Tensor5D(Tensor5D&& other)
    : data(other.data), dim1(other.dim1), dim2(other.dim2),
      dim3(other.dim3), dim4(other.dim4), dim5(other.dim5)
  {
    // if we do not change this to nullptr, destructing this and other will (probably)
    // lead to a double free.
    other.data = nullptr;
  }
  // Move assignment
  Tensor5D& operator=(Tensor5D&&) =delete;
};

// Implements a four dimensional tensor for basic types.
template <typename T>
class Tensor4D {
public:
  T* data;
  const size_t dim1;
  const size_t dim2;
  const size_t dim3;
  const size_t dim4;

  // Construct a new four dimensional tensor.
  // Input:
  //  dim{i}: the size of the i-th dimension
  //  zero: if true, initializes the memory with zero.
  Tensor4D(const size_t dim1, const size_t dim2, const size_t dim3, const size_t dim4,
           const bool zero)
    : data(zero ? alloc::calloc<T>(dim1 * dim2 * dim3 * dim4)
	   : alloc::alloc<T>(dim1 * dim2 * dim3 * dim4)),
      dim1(dim1), dim2(dim2), dim3(dim3), dim4(dim4) {}

  T get(const size_t i, const size_t j, const size_t k, const size_t l) const {
    assert(i < dim1);
    assert(j < dim2);
    assert(k < dim3);
    assert(l < dim4);

    return data[(i * (dim2 * dim3 * dim4)) + (j * (dim3 * dim4)) + (k * dim4) + l];
  }

  void set(const T value, const size_t i, const size_t j, const size_t k, const size_t l) {
    assert(i < dim1);
    assert(j < dim2);
    assert(k < dim3);
    assert(l < dim4);

    data[(i * (dim2 * dim3 * dim4)) + (j * (dim3 * dim4)) + (k * dim4) + l] = value;
  }

  // Destructor
  ~Tensor4D() {
    if (data != nullptr) {
      alloc::free(data);
    }
  }

  // Default constructor
  Tensor4D() =delete;
  // Copy constructor
  Tensor4D(const Tensor4D&) =delete;
  // Copy assignment
  Tensor4D& operator=(const Tensor4D&) =delete;
  // Move constructor
  Tensor4D(Tensor4D&& other)
    : data(other.data), dim1(other.dim1), dim2(other.dim2),
      dim3(other.dim3), dim4(other.dim4)
  {
    other.data = nullptr;
  }
  // Move assignment
  Tensor4D& operator=(Tensor4D&&) =delete;
};

// Implements a two dimensional tensor for basic types.
template <typename T>
class Tensor2D {
public:
  T* data;
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

  T get(const size_t i, const size_t j) const {
    assert(i < dim1);
    assert(j < dim2);

    return data[i * dim2 + j];
  }

  void set(const T value, const size_t i, const size_t j) {
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
  Tensor2D() =delete;
  // Copy constructor
  Tensor2D(const Tensor2D&) =delete;
  // Copy assignment
  Tensor2D& operator=(const Tensor2D&) =delete;
  // Move constructor
  Tensor2D(Tensor2D&& other)
    : data(other.data), dim1(other.dim1), dim2(other.dim2)
  {
    other.data = nullptr;
  }
  // Move assignment
  Tensor2D& operator=(Tensor2D&) =delete;
};

// Implements a one dimensional tensor for basic types.
template <typename T>
class Tensor1D {
public:
  T* data;
  const size_t size;

  // Construct a new one dimensional tensor.
  // Input:
  //  size: the size of the tensor
  //  zero: if true, initializes the memory with zero.
  Tensor1D(const size_t size, const bool zero)
    : data(zero ? alloc::calloc<T>(size) : alloc::alloc<T>(size)),
      size(size) {}

  T get(const size_t i) const {
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
  Tensor1D() =delete;
  // Copy constructor
  Tensor1D(const Tensor1D&) =delete;
  // Copy assignment
  Tensor1D& operator=(const Tensor1D&) =delete;
  // Move constructor
  Tensor1D(Tensor1D&& other)
    : data(other.data), size(other.size)
  {
    other.data = nullptr;
  }
  // Move assignment
  Tensor1D& operator=(Tensor1D&&) =delete;
};
