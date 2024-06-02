#pragma once

#include "common.hpp"
#include "tensor.hpp"
#include "tensor_macros0.hpp"

#include <cstdint>
#include <string>
#include <vector>

using namespace std;

typedef Tensor4D<float> (*ConvFunc)(const Tensor4D<float> &,
                                    const Tensor1D<float> &, const size_t,
                                    const size_t, const Tensor5D<int64_t> &,
                                    const size_t, const size_t, float);

class Implementation {
public:
  string name;
  DataOrder data_order;
  ConvFunc fn;

  Implementation(string name, DataOrder data_shape, ConvFunc func);
};

class Registry {
private:
  vector<Implementation> impls;

public:
  Registry();

  void add(Implementation impl);
  vector<Implementation> implementations();
};
