"""Op counts for bitwise GEMM."""

from math import ceil
from typing import Tuple
from plot.dimensions import output_dims
from plot.utils import BITS, POPCNT_OPS

def tnn_gemm_count(m: int, n: int, k: int) -> Tuple[int, int]:
    k_bits = k*BITS

    iops = 0
    flops = 0

    # p1 = a[..] ^ b[..]
    iops += m * n * (k_bits/BITS)
    # p2 = a[..] & b[..]
    iops += m * n * (k_bits/BITS)
    # p1 & p2
    iops += m * n * (k_bits/BITS)
    # 2x popcnt
    iops += 2 * POPCNT_OPS * (m * n * (k_bits/BITS))
    # 2x +
    iops += 2 * (m * n * (k_bits/BITS))

    return (ceil(iops), ceil(flops))

def tbn_gemm_count(m: int, n: int, k: int) -> Tuple[int, int]:
    iops = 0
    flops = 0

    # a[..] ^ b[..]
    iops += m * n * k
    # p1 & p2
    iops += m * n * k
    # 2x popocnt
    iops += 2 * POPCNT_OPS * (m * n * k)
    # 2x +
    iops += 2 * (m * n * k)
    # 2x -
    iops += 2 * (m * n)

    return (ceil(iops), ceil(flops))

def btn_gemm_count(m: int, n: int, k: int) -> Tuple[int, int]:
    iops = 0
    flops = 0

    # a[..] ^ b[..]
    iops += m * n * k
    # p1 & b[..]
    iops += m * n * k
    # popocnt
    iops += POPCNT_OPS * (m * n * k)
    # cntp2 + popcnt(..)
    iops += 2 * (m * n * k)
    # 2x -
    iops += 2 * (m * n)

    return (ceil(iops), ceil(flops))


def bnn_gemm_count(m: int, n: int, k: int) -> Tuple[int, int]:
    iops = 0
    flops = 0

    # a[..] ^ b[..]
    iops += m * n * k
    # popcnt
    iops += POPCNT_OPS * (m * n * k)
    # cntp1 + popcnt(..)
    iops += m * n * k
    # 2x -
    iops += 2 * (m * n)

    return (ceil(iops), ceil(flops))
