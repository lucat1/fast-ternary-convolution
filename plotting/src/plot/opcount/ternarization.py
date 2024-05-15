"""Op count for ternarization."""

from plot.utils import CNTBITS
from math import ceil
from typing import Tuple

def ternarize_w(
        kernel_number: int,
        chan: int,
        kernel_height: int,
        kernel_width: int) -> Tuple[int, int]:
    """Total operation count for ternarize_NCHW_to_NHWCB."""
    pri_channel = chan / CNTBITS

    iops = 0
    flops = 0

    # NOTE: probably we can ignore as it's a constant number of them
    # onebit[i] = one << i;
    iops += CNTBITS
    # currentx > quant_threshold[in]
    flops += kernel_number * kernel_height * kernel_width * pri_channel * CNTBITS
    # p2 = p2 | onebit[bit];
    iops += .5 * kernel_number * kernel_height * kernel_width * pri_channel * CNTBITS
    # currentx < (-quant_threshold[in])
    flops += .5 * kernel_number * kernel_height * kernel_width * pri_channel * CNTBITS
    # p1 = p1 | onebit[bit];
    # p2 = p2 | onebit[bit];
    iops += .25 * 2 * kernel_number * kernel_height * kernel_width * pri_channel * CNTBITS
    # TODO: accout for assignments
    # qx[..] = p1
    # qx[..] = p2

    # currentx > quant_threshold[in]
    flops += kernel_number * kernel_height * kernel_width * (chan % CNTBITS)
    # p2 = p2 | onebit[bit];
    iops += .5 * kernel_number * kernel_height * kernel_width * (chan % CNTBITS)
    # currentx < (-quant_threshold[in])
    flops += .5 * kernel_number * kernel_height * kernel_width * (chan % CNTBITS)
    # p1 = p1 | onebit[bit];
    # p2 = p2 | onebit[bit];
    iops += .25 * 2 * kernel_number * kernel_height * kernel_width * (chan % CNTBITS)
    # TODO: accout for assignments
    # qx[..] = p1
    # qx[..] = p2

    return (ceil(iops), ceil(flops))

def ternarize_q(
        kernel_number: int,
        chan: int,
        kernel_height: int,
        kernel_width: int) -> int:
    pri_channel = chan / CNTBITS

    byts = 0
    # onebit[i]
    byts += 8 * CNTBITS

    # input[..], quant_threshold[..], onebit[..]
    byts += (4+4+8) * (kernel_number * kernel_height * kernel_width * pri_channel * CNTBITS)
    # 2x qx[..]
    byts += (8+8) * (kernel_number * kernel_height * kernel_width * pri_channel)
    # input[..], quant_threshold[..]
    byts += (4+4) * (kernel_number * kernel_height * kernel_width * (chan % CNTBITS))
    # 2x qx[..]
    byts += (8+8) * (kernel_number * kernel_height * kernel_width * (chan % CNTBITS))

    return int(byts)
