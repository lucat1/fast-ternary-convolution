#pragma once

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>

// Could have been a class, but C++ doesn't allow templated methods
namespace alloc {

using namespace std;

static constexpr size_t ALIGNMENT = 32;

inline size_t __make32(size_t needed_size, size_t alignment) {
  double number_of_blocks = needed_size / (double)alignment;
  size_t alloc_size = ceil(number_of_blocks) * alignment;
  assert(alloc_size >= needed_size);
  return alloc_size;
}

template <typename T> T *alloc(size_t n) {
  T *arr = static_cast<T *>(
      aligned_alloc(ALIGNMENT, __make32(n * sizeof(T), ALIGNMENT)));
  assert(arr != nullptr);
  return arr;
}

template <typename T> T *calloc(size_t n) {
  T *arr = static_cast<T *>(
      aligned_alloc(ALIGNMENT, __make32(n * sizeof(T), ALIGNMENT)));
  assert(arr != nullptr);
  memset(arr, 0, n * sizeof(T));
  return static_cast<T *>(arr);
}

template <typename T> T *const_vec(size_t n, T val) {
  T *arr = alloc<T>(n);
  for (size_t i = 0; i < n; ++i) {
    arr[i] = val;
  }
  return arr;
};

} // namespace alloc
