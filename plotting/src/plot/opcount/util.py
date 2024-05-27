"""Total op counts."""

from typing import Tuple
from plot.datatypes import Function
from plot.opcount.activation import prelu_w
from plot.opcount.binarization import binarize_count
from plot.opcount.ternarization import ternarize_w, ternarize_q
from plot.opcount.activation import prelu_q
from plot.opcount.img2row import img2row_q
from plot.opcount.convolutions import (
    tnn_gemm_w, tnn_gemm_q,
    btn_gemm_count, bnn_gemm_count
)
from plot.opcount.img2row import img2row_w
from plot.opcount.conv import conv_w, conv_q
from plot.dimensions import packed_input_dims, output_dims
from plot.utils import BITS
import pandas as pd


def get_work_for_function(benchmark_info: pd.Series) -> Tuple[int, int]:
    """Get total op count for each function."""
    packed_dims = packed_input_dims(benchmark_info)
    out_dims = output_dims(benchmark_info)
    func = benchmark_info["fn"]

    if func == Function.ALLOC.value:
        #TODO: Find out ops for allocations
        return (1,1)

    if func == Function.FREE.value:
        #TODO: Find out ops for freeing
        return (1,1)

    if func == Function.TERNARIZE.value:
        return ternarize_w(
            benchmark_info.batch_size,
            benchmark_info.channels,
            benchmark_info.input_height,
            benchmark_info.input_width)

    if func == Function.IM2ROW.value:
        return img2row_w(
                benchmark_info.batch_size,
                packed_dims.channels * BITS,
                packed_dims.height,
                packed_dims.width,
                benchmark_info.kernel_height,
                benchmark_info.kernel_width,
                benchmark_info.stride_size)

    if func == Function.TERNA2ROW.value:
        iops1, flops1 = ternarize_w(
            benchmark_info.batch_size,
            benchmark_info.channels,
            benchmark_info.input_height,
            benchmark_info.input_width)
        iops2, flops2 = img2row_w(
                benchmark_info.batch_size,
                packed_dims.channels * BITS,
                packed_dims.height,
                packed_dims.width,
                benchmark_info.kernel_height,
                benchmark_info.kernel_width,
                benchmark_info.stride_size)
        return (iops1+iops2, flops1+flops2)

    if func == Function.BINARIZE.value:
        return binarize_count(
                benchmark_info.batch_size,
                benchmark_info.channels,
                benchmark_info.input_height,
                benchmark_info.input_width)

    if func == Function.PRELU.value:
        return prelu_w(
            benchmark_info.batch_size,
            benchmark_info.kernel_number,
            out_dims.height,
            out_dims.width)

    if func == Function.GEMM.value:
        return tnn_gemm_w(
            benchmark_info.batch_size * out_dims.height * out_dims.width,
            benchmark_info.kernel_number,
            packed_dims.channels * benchmark_info.kernel_height * benchmark_info.kernel_width)

    # if func == Function.BTN_GEMM.value:
    #     return btn_gemm_count(
    #             benchmark_info.batch_size * out_dims.height * out_dims.width,
    #             benchmark_info.kernel_number,
    #             packed_dims.channels * benchmark_info.kernel_height * benchmark_info.kernel_width)
    #
    # if func == Function.TBN_GEMM.value:
    #     return tbn_gemm_count(
    #         benchmark_info.batch_size * out_dims.height * out_dims.width,
    #         benchmark_info.kernel_number,
    #         packed_dims.channels * benchmark_info.kernel_height * benchmark_info.kernel_width)
    #
    # if func == Function.BNN_GEMM.value:
    #     return bnn_gemm_count(
    #         benchmark_info.batch_size * out_dims.height * out_dims.width,
    #         benchmark_info.kernel_number,
    #         packed_dims.channels * benchmark_info.kernel_height * benchmark_info.kernel_width)

    if func == Function.CONV.value:
        return conv_w(benchmark_info)
    
    raise ValueError(f"Got unknown function type {func}")

def get_data_movement_for_function(benchmark_info: pd.Series) -> int:
    func = benchmark_info["fn"]
    out_dims = output_dims(benchmark_info)
    packed_dims = packed_input_dims(benchmark_info)

    if func == Function.ALLOC.value or func == Function.ALLOC2.value:
        return 1

    if func == Function.FREE.value:
        return 1

    if func == Function.TERNARIZE.value:
        return ternarize_q(
            benchmark_info.batch_size,
            benchmark_info.channels,
            benchmark_info.input_height,
            benchmark_info.input_width)

    if func == Function.IM2ROW.value:
        return img2row_q(
            benchmark_info.batch_size,
            benchmark_info.channels,
            benchmark_info.input_height,
            benchmark_info.input_width,
            benchmark_info.padding_size,
            benchmark_info.padding_size,
            benchmark_info.kernel_height,
            benchmark_info.kernel_width,
            benchmark_info.stride_size,
            benchmark_info.stride_size
        )

    if func == Function.TERNA2ROW.value:
        return ternarize_q(
            benchmark_info.batch_size,
            benchmark_info.channels,
            benchmark_info.input_height,
            benchmark_info.input_width) + img2row_q(
            benchmark_info.batch_size,
            benchmark_info.channels,
            benchmark_info.input_height,
            benchmark_info.input_width,
            benchmark_info.padding_size,
            benchmark_info.padding_size,
            benchmark_info.kernel_height,
            benchmark_info.kernel_width,
            benchmark_info.stride_size,
            benchmark_info.stride_size
        )
    if func == Function.PRELU.value:
        return prelu_q(
            benchmark_info.batch_size,
            out_dims.height,
            out_dims.width, 
            benchmark_info.kernel_number)

    if func == Function.GEMM.value:
        return tnn_gemm_q(
            benchmark_info.batch_size * out_dims.height * out_dims.width,
            benchmark_info.kernel_number,
            packed_dims.channels * benchmark_info.kernel_height * benchmark_info.kernel_width)

    if func == Function.CONV.value:
        return conv_q(benchmark_info)

    return 1

    # if func == Function.BINARIZE.value:
    #     return binarize_count(
    #             benchmark_info.batch_size,
    #             benchmark_info.channels,
    #             benchmark_info.input_height,
    #             benchmark_info.input_width)
    #
    # if func == Function.BTN_GEMM.value:
    #     return btn_gemm_count(
    #             benchmark_info.batch_size * out_dims.height * out_dims.width,
    #             benchmark_info.kernel_number,
    #             packed_dims.channels * benchmark_info.kernel_height * benchmark_info.kernel_width)
    #
    # if func == Function.TBN_GEMM.value:
    #     return tbn_gemm_count(
    #         benchmark_info.batch_size * out_dims.height * out_dims.width,
    #         benchmark_info.kernel_number,
    #         packed_dims.channels * benchmark_info.kernel_height * benchmark_info.kernel_width)
    #
    # if func == Function.BNN_GEMM.value:
    #     return bnn_gemm_count(
    #         benchmark_info.batch_size * out_dims.height * out_dims.width,
    #         benchmark_info.kernel_number,
    #         packed_dims.channels * benchmark_info.kernel_height * benchmark_info.kernel_width)
    #
    # if func == Function.CONV.value:
    #     return conv_op_count(benchmark_info)
