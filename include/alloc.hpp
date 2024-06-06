#pragma once

#include "measure.hpp"
#include <cassert>
#include <cstdlib>
#include <cstring>

#define ALIGN

// Could have been a class, but C++ doesn't allow templated methods
namespace alloc {

using namespace std;

static constexpr size_t ALIGNMENT = 64;

#define alloc_size(n) (n * sizeof(T) + ALIGNMENT + sizeof(size_t))

template <typename T> T *alloc(size_t n) {
  size_t bytes = alloc_size(n);
  Measure::get_instance()->track_memory(bytes);

  measure_point(measurement_point::alloc, MeasurementEvent::START);
#ifdef ALIGN
  size_t raw_addr = (size_t)malloc(bytes);
  assert(raw_addr != (size_t) nullptr);
  size_t *aligned_addr = static_cast<size_t *>((void *)raw_addr);
  if ((raw_addr + sizeof(size_t)) % ALIGNMENT > 0)
    aligned_addr = static_cast<size_t *>(static_cast<void *>(
        (void *)(raw_addr +
                 (ALIGNMENT - ((raw_addr + sizeof(size_t)) % ALIGNMENT)))));

  size_t *original_addr_ptr = &aligned_addr[0];
  T *data_ptr = static_cast<T *>(static_cast<void *>(&aligned_addr[1]));
  assert((size_t)data_ptr % ALIGNMENT == 0);

  *original_addr_ptr = (size_t)raw_addr;
#else
  T *data_ptr = static_cast<T *>(malloc(bytes));
  assert(data_ptr != nullptr);
#endif
  measure_point(measurement_point::alloc, MeasurementEvent::END);
  return data_ptr;
}

template <typename T> T *calloc(size_t n) {
  T *arr = alloc<T>(n);
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

template <typename T> void free(T *aligned_addr) {
#ifdef ALIGN
  size_t *memory_area =
      static_cast<size_t *>(static_cast<void *>(aligned_addr));
  T *original_addr_ptr = static_cast<T *>((void *)memory_area[-1]);
  std::free(original_addr_ptr);
#else
  std::free(aligned_addr);
#endif
}

} // namespace alloc
