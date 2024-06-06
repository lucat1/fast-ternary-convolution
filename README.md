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

## Implementation overview
- **original**: original implementation using vectors (nchw)
- **nchw**: simple implementation using tensors (nchw)
- **nhwc**: simple implementation using tensors (nhwc)
- **ternary_nhwc**: nhwc using ternary operators for prelu/ternarize
  - overall seems to be slightly worse
  - seems to be slightly worse in ternarize, maybe sometimes a tiny bit faster in prelu
- **nchw_tmacro1**: replace getters and setters of **t**ensors with simple macros
- **nchw_tmacro2**: manually eliminate redundant computation in indicies
- **nchw_tmacro1_sinline**: use inline keyword for **s**teps like ternarize, gemm, prelu, etc.
- **nchw_tmacro2_sinline**: same as above, but for tmacro2
- **nhwc_tmacro1**: nhwc order
- **nhwc_tmacro2**: nhwc order
- **nhwc_tmacro1_sinline**: nhwc order
- **nhwc_tmacro2_sinline**: nhwc order
- **indirect**: essentially merge im2row + gemm? @Luca please double check
- **more_indirect**: @Luca please fill out
  - possible optimization: merge with ternarize? precompute statically?
  - does this have any benefits besides performance? mention that in report?
- **tern2row**: naively merge ternarize and im2row
  - leads to massive slowdown due to unnecessary recomputation
- **tern2row_cpy**: avoid recomputation by copying already computed elements
  - uses a loop for copying, has the edge of memcpy in daniel_plots*.csv
  - seems to be very slightly better than not merging
- **tern2row_memcpy**: copy using memcpy instead of a loop
  - loses to a loop, maybe because memcpy adds overhead (and we don't copy a lot of data at once?)
- **t2r_gemmLU**: merge gemm and PreLU (based on tern2row_cpy)
- **t2r_gemmLU_lord**: conditionally swaps the loop order
  - reasoning: slides on model ATLAS: want to reuse the smaller matrix
  - just naively switching to N-M makes it worse
  - M<N=> N-M: whenever this branch is taken (if batch_size * oh * ow < kn according to Luca), performance seems to get a bit worse
  - Why? Unsure. Maybe it has to do with how the second matrix is passed in? i.e. N-K instead K-N? (not sure about this)
- **t2r_gemmLU_block**: Block gemmLU
  - we should ideally have autotune to find the blocking parameters
- **t2r_gemmLU_unroll**: Unroll the most inner loop in SSA style
  - seems to be worse than block
  - hypothesis: by unrolling manually, we take away freedom from compiler to optimize it
  - suscpicions further confirmed by disabling loop-unrolling via compiler flag: block and unroll appear to be similarly bad (although rerun this and confirm)
  - we should check the assembly for the report
## Running vectorized Code
Some parts of the code have been vectorized (currently only with AVX512). To compile the code, please use the following compiler flags
```
OPTFLAGS = -march=native -O3 -fno-tree-vectorize -std=c++20 -mavx512f -mavx512vpopcntdq
```
