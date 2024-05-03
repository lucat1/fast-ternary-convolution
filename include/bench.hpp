#pragma once

#include "common.hpp"
#include "impl.hpp"

#include <vector>

void bench(Registry r, vector<InfraParameters> *params);
void measure_overhead();
