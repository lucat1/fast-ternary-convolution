"""Op count for activation."""

from math import ceil
from typing import Tuple


def PreLU_f32_compare_count(n: int, c: int, h: int, w: int) -> int:
    """Compute the number of f32 compares for PreLU."""
    return n * c * h * w


def PreLU_f32_mul_count(n: int, c: int, h: int, w: int) -> int:
    """Get the approximate number of muls in PreLU."""
    return n * c * h * w // 2


def PreLU_assign_count(n: int, c: int, h: int, w: int) -> int:
    """Compute the number of f32 compares for PreLU."""
    return n * c * h * w


def prelu_count(n: int, c: int, h: int, w: int) -> Tuple[int, int]:
    """Total operation count for activation."""
    iops = 0
    flops = 0

    # current > 0
    flops += n * c * h * w
    # current * alpha
    flops += .5 * n * c * h * w

    return (ceil(iops), ceil(flops))
