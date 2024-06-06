from codegen.straightforward import gemm_kernel_macro as straight_macro
from codegen.unrolled import gemm_kernel_macro as unrolled_macro
from codegen.vector import gemm_kernel_macro_256 as vector_256, gemm_kernel_macro_512 as vector_512, gemm_kernel_macro_256_libpopcnt as vector_256_popcnt
from codegen.vector_unrolled import gemm_kernel_macro_256 as unrolled_256, gemm_kernel_macro_512 as unrolled_512, gemm_kernel_macro_256_libpopcnt as unrolled_256_popcnt

if __name__ == "__main__":
    # code = straight_macro()
    # code = unrolled_macro(8)
    # code = unrolled_macro(4)
    # code = vector_256()
    # code = unrolled_256(2, 2)
    # code = vector_512()
    # code = unrolled_512(4, 4)
    code = unrolled_512(2, 2)

    # code = vector_256_popcnt()
    # code = unrolled_256_popcnt(2, 2)

    print(code.gen())
    # only for debugging with python3 -m codegen | vi
    print("// vi: ft=c")
