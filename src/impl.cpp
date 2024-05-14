#include "impl.hpp"

Implementation::Implementation(string name, DataOrder data_order, ConvFunc func)
    : name(name), data_order(data_order), fn(func) {}

Registry::Registry() {}

void Registry::add(Implementation impl) { impls.push_back(impl); }

vector<Implementation> Registry::implementations() { return impls; }
