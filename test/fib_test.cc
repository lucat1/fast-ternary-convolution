#include "gtest/gtest.h"
#include <iostream>

#include "asl/fib.hh"

TEST(Fib, LowNumbers) {
  EXPECT_EQ(0, asl::fib(0));
  EXPECT_EQ(1, asl::fib(1));
  EXPECT_EQ(1, asl::fib(2));
  EXPECT_EQ(2, asl::fib(3));
}
