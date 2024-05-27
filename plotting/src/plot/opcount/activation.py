"""Op count for activation."""

from math import ceil
from typing import Tuple

def prelu_q(n: int, out_h: int, out_w: int, c: int) -> int:
    iter = n*out_h*out_w*c
    # there's one read and one write (counts double)
    return 4*3*iter

def prelu_w(n: int, c: int, out_h: int, out_w: int) -> Tuple[int, int]:
    """Total operation count for activation."""
    iops = 0
    flops = 0

    # current > 0
    flops += n * c * out_h * out_w
    # current * alpha
    flops += .5 * n * c * out_h * out_w

    return (ceil(iops), ceil(flops))
