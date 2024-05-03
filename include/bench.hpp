#pragma once

#include "common.hpp"
#include "impl.hpp"

#include <vector>

void bench(Registry r, vector<InfraParameters> *params, string output);
void measure_overhead();
