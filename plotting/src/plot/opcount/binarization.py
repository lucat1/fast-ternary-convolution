"""Op counts for binarization."""

from plot.utils import CNTBITS
from math import ceil
from typing import Tuple

def binarize_count(
        batch_size: int,
        channels: int,
        input_height: int,
        input_width: int) -> Tuple[int, int]:
    """Total operation count for binarize_NCHW_to_NHWC."""
    pri_channel = channels / CNTBITS

    iops = 0
    flops = 0

    # NOTE: probably we can ignore as it's a constant number of them
    # onebit[i] = one << i;
    iops += CNTBITS
    # input[..] < quant_threshold[in]
    flops += batch_size * input_height * input_width * pri_channel * CNTBITS
    # p1 = p1 | onebit[bit];
    iops += .5 * batch_size * input_height * input_width * pri_channel * CNTBITS
    # TODO: handle assignment
    # qx[..] = p1

    # input[..] < quant_threshold[in]
    iops += batch_size * input_height * input_width * (channels % CNTBITS)
    # NOTE: account for assignments
    # qx[..] = p1
    return (ceil(iops), ceil(flops))
