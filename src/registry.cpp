#include "registry.hpp"

#include <cassert>
#include <cstdlib>
#include <map>

namespace registry {

std::map<name_t, func_t> entries;

void define(std::map<name_t, func_t> map) { entries = map; }

std::map<name_t, func_t>::iterator begin() { return begin(entries); }
std::map<name_t, func_t>::iterator end() { return end(entries); }

} // namespace registry
