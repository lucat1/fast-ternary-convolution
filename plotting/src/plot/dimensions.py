"""Helper module to compute dimensions."""

from plot.utils import (CNTBITS, BITS)
from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass
class PackedDims:
    """Packed data dimensions used in tnn computation."""
    height: int
    width: int
    channels: int


def packed_input_dims(params: pd.Series) -> PackedDims:
    """Get packed data dimensions."""
    packed_height = params.input_height + 2 * params.padding_size
    packed_width = params.input_width + 2 * params.padding_size
    if params.channels % CNTBITS != 0:
        packed_channels = params.channels // CNTBITS + 1
    else:
        packed_channels = params.channels // CNTBITS
    return PackedDims(packed_height, packed_width, packed_channels)

@dataclass
class OutputDims:
    """Output data dimensions."""

    height: int
    width: int


def output_dims(params: pd.Series) -> OutputDims:
    """Get output data dimensions."""
    packed_dims = packed_input_dims(params)
    output_height = (packed_dims.height - params.kernel_height + 1) // params.stride_size
    output_width = (packed_dims.width - params.kernel_width + 1) // params.stride_size
    return OutputDims(output_height, output_width)

@dataclass
class FusedDims:
    """Fused data dimensions."""

    height: int
    width: int


def fused_dims(params: pd.Series) -> FusedDims:
    """Get fused data dimensions."""
    od = output_dims(params)
    pd = packed_input_dims(params)
    fused_height = od.height * od.width
    fused_width = params.kernel_height * params.kernel_width * (pd.channels * BITS);
    return FusedDims(fused_height, fused_width)
