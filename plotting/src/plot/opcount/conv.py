"""Overall operation count of conv."""

from typing import Tuple
import pandas as pd
from plot.dimensions import packed_input_dims, output_dims
from plot.datatypes import ConvType
from plot.opcount.binarization import binarize_count
from plot.opcount.ternarization import ternarize_w, ternarize_q
from plot.opcount.convolutions import (
    tnn_gemm_w, tnn_gemm_q,
    tbn_gemm_count,
    bnn_gemm_count,
    btn_gemm_count)
from plot.opcount.img2row import img2row_w, img2row_q
from plot.utils import BITS
from plot.opcount.activation import prelu_w, prelu_q

def conv_q(benchmark_info: pd.Series) -> int:
    """Get the overall op count of conv."""
    pack_dim = packed_input_dims(benchmark_info)
    out_dim = output_dims(benchmark_info)

    byt = 0

    # Quantize and Convolutions.
    match benchmark_info.ct:
        case ConvType.TNN:
            byt += ternarize_q(
                benchmark_info.batch_size,
                benchmark_info.channels,
                benchmark_info.input_height,
                benchmark_info.input_width)
            byt += img2row_q(
                benchmark_info.batch_size,
                benchmark_info.channels,
                benchmark_info.input_height,
                benchmark_info.input_width,
                benchmark_info.padding_size,
                benchmark_info.padding_size,
                benchmark_info.kernel_height,
                benchmark_info.kernel_width,
                benchmark_info.stride_size,
                benchmark_info.stride_size)

    # Bitwise GEMM
    gemm_iops = 0
    gemm_flops = 0
    match benchmark_info.ct:
        case ConvType.TNN:
            byt += tnn_gemm_q(
                benchmark_info.batch_size * out_dim.height * out_dim.width,
                benchmark_info.kernel_number,
                pack_dim.channels * benchmark_info.kernel_height * benchmark_info.kernel_width)

    # Activation.
    byt += prelu_q(
        benchmark_info.batch_size,
        benchmark_info.kernel_number,
        out_dim.height,
        out_dim.width)

    return byt

def conv_w(benchmark_info: pd.Series) -> Tuple[int, int]:
    """Get the overall op count of conv."""
    pack_dim = packed_input_dims(benchmark_info)
    out_dim = output_dims(benchmark_info)

    iops = 0
    flops = 0

    # Quantize and Convolutions.
    match benchmark_info.ct:
        case (ConvType.TNN | ConvType.TBN):
            ternarize_iops, ternarize_flops = ternarize_w(
                benchmark_info.batch_size,
                benchmark_info.channels,
                benchmark_info.input_height,
                benchmark_info.input_width)
            iops += ternarize_iops
            flops += ternarize_flops

            img2row_iops, img2row_flops = img2row_w(
                benchmark_info.batch_size,
                pack_dim.channels * BITS,
                pack_dim.height,
                pack_dim.width,
                benchmark_info.kernel_height,
                benchmark_info.kernel_width,
                benchmark_info.stride_size)
            iops += img2row_iops
            flops += img2row_flops

        case (ConvType.BTN | ConvType.BNN):
            binarize_iops, binarize_flops = binarize_count(
                benchmark_info.batch_size,
                benchmark_info.channels,
                benchmark_info.input_height,
                benchmark_info.input_width)
            iops += binarize_iops
            flops += binarize_flops

            img2row_iops, img2row_flops = img2row_w(
                benchmark_info.batch_size,
                pack_dim.channels,
                pack_dim.height,
                pack_dim.width,
                benchmark_info.kernel_height,
                benchmark_info.kernel_width,
                benchmark_info.stride_size)
            iops +=  img2row_iops
            flops += img2row_flops

    # Bitwise GEMM
    gemm_iops = 0
    gemm_flops = 0
    match benchmark_info.ct:
        case ConvType.TNN:
            gemm_iops, gemm_flops = tnn_gemm_w(
                benchmark_info.batch_size * out_dim.height * out_dim.width,
                benchmark_info.kernel_number,
                pack_dim.channels * benchmark_info.kernel_height * benchmark_info.kernel_width)

        case ConvType.TBN:
            gemm_iops, gemm_flops = tbn_gemm_count(
                benchmark_info.batch_size * out_dim.height * out_dim.width,
                benchmark_info.kernel_number,
                pack_dim.channels * benchmark_info.kernel_height * benchmark_info.kernel_width)

        case ConvType.BTN:
            gemm_iops, gemm_flops = btn_gemm_count(
                benchmark_info.batch_size * out_dim.height * out_dim.width,
                benchmark_info.kernel_number,
                pack_dim.channels * benchmark_info.kernel_height * benchmark_info.kernel_width)

        case ConvType.BNN:
            gemm_iops, gemm_flops = bnn_gemm_count(
                benchmark_info.batch_size * out_dim.height * out_dim.width,
                benchmark_info.kernel_number,
                pack_dim.channels * benchmark_info.kernel_height * benchmark_info.kernel_width)

    iops += gemm_iops
    flops += gemm_flops

    # Activation.
    prelu_iops, prelu_flops = prelu_w(
        benchmark_info.batch_size,
        benchmark_info.kernel_number,
        out_dim.height,
        out_dim.width)
    iops = prelu_iops
    flops = prelu_flops

    return (iops, flops)
