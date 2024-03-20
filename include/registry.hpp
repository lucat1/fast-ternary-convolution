#ifndef _REGISTRY_HPP
#define _REGISTRY_HPP

#include <cstddef>
#include <vector>

namespace registry {

typedef void (*func_t)(double *, double *, double *, size_t);
typedef struct entry {
  const char *name;
  func_t f;
} entry_t;

void def(const char *name, func_t f);
std::vector<entry_t *> &all();

} // namespace registry

#endif // _REGISTRY_HPP
