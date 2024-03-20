#include "registry.hpp"

#include <cassert>
#include <cstdlib>
#include <vector>

namespace registry {

std::vector<entry_t *> entries;

void def(const char *name, func_t f) {
  entry_t *e = (entry_t *)malloc(sizeof(entry_t));
  assert(e != nullptr);
  e->name = name;
  e->f = f;

  entries.push_back(e);
}

std::vector<entry_t *> &all() { return entries; }

} // namespace registry
