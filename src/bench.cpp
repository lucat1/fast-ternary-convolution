#include "bench.hpp"

#include <iomanip>
#include <iostream>

using namespace std;

void bench(Registry r) {
  for (auto impl : r.implementations()) {
    cout << setw(name_space) << impl.name << " :: "
         << " whatever benchmark" << endl;
  }
}
