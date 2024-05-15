"""Op count for img2row."""

from math import ceil
from typing import Tuple


def img2row_NHWCB_to_N_OHOW_KHKWC_assign_count(
        batch_size: int,
        num_channels: int,
        input_height: int,
        input_width: int,
        kernel_height: int,
        kernel_width: int,
        stride_size: int) -> int:
    """Get number of assignments in img2row_NHWCB_to_N_OHOW_KHKWC."""
    output_height = (input_height - kernel_height + 1) // stride_size;
    output_width = (input_width - kernel_width + 1) // stride_size;
    return batch_size * \
        output_height * \
        output_width * \
        kernel_height * \
        kernel_width * \
        num_channels


def img2row_count(
        batch_size: int,
        num_channels: int,
        input_height: int,
        input_width: int,
        kernel_height: int,
        kernel_width: int,
        stride_size: int) -> Tuple[int, int]:
    iops = 0
    flops = 0

    # This function only does assignments which we don't take into account for now

    return(ceil(iops), ceil(flops))
