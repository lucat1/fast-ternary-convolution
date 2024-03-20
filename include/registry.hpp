#ifndef _REGISTRY_HPP
#define _REGISTRY_HPP

#include <cstddef>
#include <map>
#include <utility>

namespace registry {

typedef void (*func_t)(double const *, double const *, double *, size_t);
typedef const char *name_t;

void define(std::map<name_t, func_t> entries);
std::map<name_t, func_t>::iterator begin();
std::map<name_t, func_t>::iterator end();

} // namespace registry

#endif // _REGISTRY_HPP
