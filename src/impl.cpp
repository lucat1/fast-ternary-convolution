#include "impl.hpp"

Implementation::Implementation(string name, ConvFunc func)
    : name(name), fn(func) {}

Registry::Registry() {}

void Registry::add(Implementation impl) { impls.push_back(impl); }

vector<Implementation> Registry::implementations() { return impls; }
