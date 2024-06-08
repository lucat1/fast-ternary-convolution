from typing import List
import os
from codegen.unrolled import gemm_kernel_macro as straight_unrolled
from codegen.vector_unrolled import gemm_kernel_macro_256 as avx2_unrolled, gemm_kernel_macro_512 as avx512_unrolled, gemm_kernel_macro_256_libpopcnt as avx2_libpop_unrolled
from codegen.straightforward import gemm_kernel_macro as straight
from codegen.vector import gemm_kernel_macro_256 as avx2, gemm_kernel_macro_512 as avx512, gemm_kernel_macro_256_libpopcnt as avx2_libpop

def copy_and_replace(src_dir, dest_dir, old_strings: List[str], new_strings: List[str]):
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Iterate over all files in the source directory
    for filename in os.listdir(src_dir):
        # Construct full file path
        src_file = os.path.join(src_dir, filename)
        dest_file = os.path.join(dest_dir, filename)

        # Only process files, skip directories
        if os.path.isfile(src_file):
            # Read the content of the file
            with open(src_file, 'r', encoding='utf-8') as file:
                file_content = file.read()

            # Replace the old string with the new string
            for old_string, new_string in zip(old_strings, new_strings):
                file_content = file_content.replace(old_string, new_string)

            # Write the updated content to the destination file
            with open(dest_file, 'w', encoding='utf-8') as file:
                file.write(file_content)

SIZES = [1, 2, 3, 4, 5, 6, 7, 8]
KIND = ["straight", "avx2", "avx2libpop"] # , "avx512"]
for size in SIZES:
    for kind in KIND:
        impl = f"rollin_{kind}_{size}"
        if kind == "straight":
            macro = straight_unrolled(size) if size > 1 else straight()
        elif kind == "avx2":
            macro = avx2_unrolled(size, size) if size > 1 else avx2()
        elif kind == "avx2libpop":
            macro = avx2_libpop_unrolled(size, size) if size > 1 else avx2_libpop()
        elif kind == "avx512":
            macro = avx512_unrolled(size, size) if size > 1 else avx512()
        else:
            raise NotImplementedError()

        macro = macro.gen()
        copy_and_replace("template/src", f"src/impl/{impl}", ["%impl%", "%macro%"], [impl, macro])
        copy_and_replace("template/include", f"include/impl/{impl}", ["%impl%", "%macro%"], [impl, macro])

        print(f"#include \"impl/{impl}/tab.hpp\"")
        print(f"{{\"{impl}\", DataOrder::NHWC, {impl}::conv}},")
