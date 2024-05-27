"""Op count for img2row."""

from math import ceil
from typing import Tuple


def img2row_NHWCB_to_N_OHOW_KHKWC_assign_count(
        batch_size: int,
        channels: int,
        input_height: int,
        input_width: int,
        kernel_height: int,
        kernel_width: int,
        stride_size: int) -> int:
    """Get number of assignments in img2row_NHWCB_to_N_OHOW_KHKWC."""
    output_height = (input_height - kernel_height) // stride_size  + 1;
    output_width = (input_width - kernel_width) // stride_size + 1;
    return batch_size * \
        output_height * \
        output_width * \
        kernel_height * \
        kernel_width * \
        channels

def img2row_q(
        batch_size: int,
        channels: int,
        input_h: int,
        input_w: int,
        padding_h: int,
        padding_w: int,
        kernel_h: int,
        kernel_w: int,
        stride_h: int,
        stride_w: int) -> int:
    packed_h = input_h + 2 * padding_h
    packed_h = input_w + 2 * padding_w
    pc = channels // 64 + (1 if channels % 64 > 0 else 0)
    out_h = (packed_h - kernel_h) // stride_h + 1;
    out_w = (packed_h - kernel_w) // stride_w + 1;

    count = batch_size * out_h * out_w * kernel_h * kernel_w * pc * 2
    # one i64 is read and 1 is written (counts as 2) every iteration (count)
    return 8*3*count

def img2row_w(
        batch_size: int,
        channels: int,
        input_height: int,
        input_width: int,
        kernel_height: int,
        kernel_width: int,
        stride_size: int) -> Tuple[int, int]:
    iops = 0
    flops = 0

    # This function only does assignments which we don't take into account for now

    return(ceil(iops), ceil(flops))
