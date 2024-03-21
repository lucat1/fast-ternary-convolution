#ifndef _BENCH_HPP
#define _BENCH_HPP

#include "registry.hpp"
#include "test.hpp"

namespace bench {

double measure(registry::func_t f, registry::env_t &env);

void all();

} // namespace bench

#endif // _BENCH_HPP
