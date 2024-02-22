#include "asl/fib.hh"

namespace asl {

int fib(int n) {
  if (n <= 1)
    return n;

  return fib(n - 1) + fib(n - 2);
}

} // namespace asl
