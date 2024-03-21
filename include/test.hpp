#ifndef _TEST_HPP
#define _TEST_HPP

#include "registry.hpp"

namespace test {

bool compare(registry::func_t a, registry::func_t b, registry::env_t &env);

void all();

} // namespace test

#endif // _TEST_HPP
