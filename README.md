# ASL TAB

## Running and Building

### Running TNN

To build run:
```
make
```

To view help:
```
./tnn -h
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

### Plotting

Generate benchmarks in `.csv` using any of the scripts in `scripts`.

```bash
./scripts/bench_*
```

Then to run the plotting, make sure the `.csv` file is in the `benchmarks` folder.

Then to generate performance and runtime plots run:

```bash
python3 -m plotting.plotter
```

The corresponding plots will be in the `plots` folder.

To filter implementaion data from an existing `.csv` you can filter out the implementations into a new `.csv` via:

```bash
 python3 -m plotting.filter_csv -i results/csvs/final/fc.csv -o benchmarks/fc.csv -n original best_impl_avx512
 ```

## Adding New Optimizations

To introduce a new optimization you need to:
1. Add a new directory under `include/main_impls` or `include/minor_impls`, depending on whether or not this is a major improvement or just a minor change.
2. Add a new `tab.hpp` header file to this new directory.
3. Add a new directory under `src/main_impls` or `src/minor_impls`.
4. Add a new `tab.cpp` file to his new directory.
5. Define a new namespace for this new optimization.
6. To `src/main.cpp` add a new element to `vector<Implementation> impls` where you give a name to the optimization, specify the order of the tensor dimensions, and the convolution function.

## Adding Measurements

To time a specific operation, for example ternary GEMM, surround the call of the procedure with a call to `measure_point` defined in `measure.hpp`:
```C++
measure_point(MeasurementFunction::TNN_GEMM, MeasurementEvent::START);
auto gemm_result = ternary_gemm(reshaped, kernel);
measure_point(MeasurementFunction::TNN_GEMM, MeasurementEvent::END);
```

## Implementation Overview
### Main Implementations
- **original**: original implementation using vectors (data order: nchw)
- **data_order_nhwc**: simple implementation using tensors (data order changed to nhwc)
- **data_order_nchw_tensor_macro1**: replace getters and setters of tensors with simple macros
- **t2r_gemmLU**: merge gemm and PreLU (based on tern2row_cpy)
- **best_impl_avx2**: Overall best implementation using AVX2 vectorization.
- **best_impl_avx512**: Overall best implementation using AVX512 vectorization.

### Minor Implementations

#### NCHW Data Order

- **nchw**: simple implementation using tensors (data order: nchw)
- **nchw_tmacro1**: replace getters and setters of tensors with simple macros
- **nchw_tmacro1_sinline**: use inline keyword for **s**teps like ternarize, gemm, prelu, etc.
- **nchw_tmacro2**: manually eliminate redundant computation in indices
- **nchw_tmacro2_sinline**

#### NHWC Data Order

- **nhwc_tmacro1_sinline**
- **nhwc_tmacro2**
- **nhwc_tmacro2_sinline**

##### Indirect Convolution

- **indirect**: compute an indirection buffer instead of im2row
- **more_indirect**: smaller indirection buffer

##### ternarize+im2row

- **tern2row**: naively merge ternarize and im2row
- **tern2row_cpy**: avoid recomputation by copying already computed elements (uses a loop for copying)
- **tern2row_memcpy**: copy using memcpy instead of a loop
- **t2r_ur_gemmLU_block**: unrolling by 2 in ternarize+im2row, and blocking merged gemm+prelu

##### gemm+prelu

- **t2r_gemmLU_autoblock**: template for searching for the best blocking parameters
- **t2r_gemmLU_block**: Block gemmLU
- **t2r_gemmLU_lord**: conditionally swaps the loop order (inspired by Model ATLAS)
- **t2r_gemmLU_unroll**: Unroll the most inner loop in SSA style

##### AVX2: ternarize+im2row

- **t2r_avx2u_gemmLU_block**: axv2 and unrolled by 2 cleanup loop in ternarize+im2row
- **t2r_avx2u_ur_gemmLU_block**: unrolled by 2 axv2 and unrolled by 2 cleanup loop in ternarize+im2row
- **t2r_avx2u_permute_gemmLU_block**: axv2 with permute intrinsic for reducing results and unrolled by 2 cleanup loop in ternarize+im2row
- **t2r_avx2u_permute_ur_gemmLU_block**: unrolled by 2 axv2 with permute intrinsic for reducing results and unrolled by 2 cleanup loop in ternarize+im2row

##### AVX2: gemm+prelu

- **avx2**: straightforward AVX2, unrolled twice
- **avx2_lessunpack**: same as before + more computations and less unpacks
- **avx2_lessunpack_popout**: same as before + libpopcnt on a big vector
- **avx2_popout**: straightforward AVX2 with libpopcnt on a big vector
- **t2r_gemmLU_block_avx2**: avx2 used in blocked gemmLU

##### AVX512: ternarize+im2row

- **t2r_avx512u_gemmLU_block**: avx512 and unrolled cleanup loop in ternarize+im2row
- **t2r_avx512u_ur_gemmLU_block**: unrolled by 2 avx512 and unrolled cleanup loop in ternarize+im2row

##### AVX512: gemm+prelu

- **t2r_gemmLU_block_avx512**: axv512 used in blocked gemmLU

##### Miscellaneous

- **ternary_nhwc**: nhwc using ternary operators for prelu/ternarize

## Running Vectorized Code
Some parts of the code have been vectorized with AVX512.
If you do not have AVX512 on your machine, remove all code that uses AVX512 and the corresponding flags in `Makefile`.
