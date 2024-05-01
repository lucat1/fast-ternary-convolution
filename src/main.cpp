#include "bench.hpp"
#include "impl.hpp"
#include "verify.hpp"

#include "impl/baseline/tab.hpp"

using namespace std;

int main() {
  measure_overhead();

  Registry r;
  r.add(Implementation("baseline", baseline::conv));

  // verify(r);
  bench(r);
}
