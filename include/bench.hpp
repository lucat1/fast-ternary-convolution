#ifndef _BENCH_HPP
#define _BENCH_HPP

#include "registry.hpp"
#include "test.hpp"

namespace bench {

double measure(registry::name_t name, registry::func_t f, test::env_t env);

} // namespace bench

#endif // _BENCH_HPP
