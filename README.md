# ASL TAB

## Running and building

To build run:
```
make
```

To run all tests do:
```
./tnn -t
```

To get benchmarks using particular parameters, for example from `parameters/channels.csv` run:

```
./tnn -b -p parameters/channels.csv -o benchmarks/channels.csv
```

We use `clang-format` to make sure our C and C++ code is consistently formatted.
Install `clang-format` on your system.

Before committing and pushing to the remote be sure to run
```
make format
```

Alternatively you can add a pre-commit hook to do formatting adding a file named `pre-commit` to your local `.git/hooks` directory:
```bash
#!/bin/sh
# Run the make format command to format code before committing
make format

# Check if make format succeeded
if [ $? -ne 0 ]; then
  echo "Code formatting failed. Please fix any issues and try committing again."
  exit 1
fi
```

## Adding new optimizations

To introduce a new optimization you need to:
1. Add a new directory under `include/impl`.
2. Add a new `tab.hpp` header file to this new directory.
3. Add a new directory under `src/impl`.
4. Add a new `tab.cpp` file to his new directory.
5. Define a new namespace for this new optimization.
6. To `src/main.cpp` add a new element to `vector<Implementation> impls` where you give a name to the optimization, specify the order of the tensor dimensions, and the convolution function.

## Adding measurements

To time a specific operation, for example ternary GEMM, surround the call of the procedure with a call to `measure_point` defined in `measure.hpp`:
```C++
measure_point(MeasurementFunction::TNN_GEMM, MeasurementEvent::START);
auto gemm_result = ternary_gemm(reshaped, kernel);
measure_point(MeasurementFunction::TNN_GEMM, MeasurementEvent::END);
```
