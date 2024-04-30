#pragma once
#include <cstdint>

#if defined(__GNUC__) || defined(__linux__)
#define VOLATILE __volatile__
#define ASM __asm__
#else
/* if we're neither compiling with gcc or under linux, we can hope
 * the following lines work, though probably it won't */
#define ASM asm
#define VOLATILE
#endif

/* This is the RDTSC timer.
 * RDTSC is an instruction on several Intel and compatible CPUs that Reads the
 * Time Stamp Counter. The Intel manuals contain more information.
 */

#define COUNTER_LO(a) ((a).int32.lo)
#define COUNTER_HI(a) ((a).int32.hi)
#define COUNTER_VAL(a) ((a).int64)

#define COUNTER(a) ((uint64_t)COUNTER_VAL(a))

#define COUNTER_DIFF(a, b) (COUNTER(a) - COUNTER(b))

typedef union {
  uint64_t int64;
  struct {
    uint32_t lo, hi;
  } int32;
} tsc_counter;

#define RDTSC(cpu_c)                                                           \
  ASM VOLATILE("rdtsc" : "=a"((cpu_c).int32.lo), "=d"((cpu_c).int32.hi))
#define CPUID() ASM VOLATILE("cpuid" : : "a"(0) : "bx", "cx", "dx")

static uint64_t read_start() {
  tsc_counter start;
  CPUID();
  RDTSC(start);
  return COUNTER_VAL(start);
}

static uint64_t read_end() {
  tsc_counter end;
  RDTSC(end);
  CPUID();
  return COUNTER_VAL(end);
}
