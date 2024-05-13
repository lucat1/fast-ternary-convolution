#pragma once

#include "common.hpp"
#include "tensor.hpp"

#include <cstdint>
#include <string>
#include <vector>

using namespace std;

typedef Tensor4D<float> (*ConvFunc)(const Tensor4D<float> &,
                                    const Tensor1D<float> &, const size_t,
                                    const size_t, const Tensor5D<int64_t> &,
                                    const size_t, const size_t, float);

typedef Tensor5D<int64_t> (*Ternarize)(const Tensor4D<float> &,
                                       const Tensor1D<float> &, const size_t,
                                       const size_t);

class Implementation {
public:
  string name;
  DataOrder data_order;
  ConvFunc fn;
  Ternarize ternarize;

  Implementation(string name, DataOrder data_shape, ConvFunc func,
                 Ternarize ternarize);
};

class Registry {
private:
  vector<Implementation> impls;

public:
  Registry();

  void add(Implementation impl);
  vector<Implementation> implementations();
};
