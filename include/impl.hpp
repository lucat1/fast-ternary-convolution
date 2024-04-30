#pragma once

#include "common.hpp"

#include <cstdint>
#include <string>
#include <vector>

using namespace std;

typedef void (*ConvFunc)(ConvolutionType, int *, float *, uint32_t, uint32_t,
                         uint32_t, uint32_t, float *, int, int64_t *, uint32_t,
                         uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                         float, float *);

class Implementation {
public:
  string name;
  ConvFunc fn;

  Implementation(string name, ConvFunc func);
};

class Registry {
private:
  vector<Implementation> impls;

public:
  Registry();

  void add(Implementation impl);
  vector<Implementation> implementations();
};
