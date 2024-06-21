#pragma once

#include "common.hpp"
#include "impl.hpp"

#include <vector>

void bench(Registry r, vector<InfraParameters> *params, string output,
           bool convonly, bool hot_cache);
void measure_overhead();
