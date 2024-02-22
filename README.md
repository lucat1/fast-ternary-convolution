# Requirements

To setup your development environment, you'll need:
- CMake >= 3.16
- a C/C++ compiler supported by CMake
- Git (to clone GTest and GBenchmark)

# Building

To build the project, you should follow the usual CMake steps. Here's a gist of it:

```sh
mkdir build && cd build # Creates and moves you into the build directory
cmake .. # Generates the makefiles from the CMake definitions
make # Compiles the code along with the test and benchmarking suites
```

You can now run the tests and benchmarks by using:
```sh
make test
./bench/bench_asl
```

# Credits

The basis for the project structure is heavily borrowed from [Philippe
Desjardins-Proulx's boilerplate](https://github.com/PhDP/cmake-gtest-gbench-starter)
