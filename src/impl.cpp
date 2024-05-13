#include "impl.hpp"

Implementation::Implementation(string name, DataOrder data_shape, ConvFunc func)
    : name(name), data_shape(data_shape), fn(func) {}

Registry::Registry() {}

void Registry::add(Implementation impl) { impls.push_back(impl); }

vector<Implementation> Registry::implementations() { return impls; }
