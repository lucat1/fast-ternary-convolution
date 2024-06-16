"""Compile data points from csv files."""

from dataclasses import dataclass
from plotting.opcount.op_counts import compute_ops
from plotting.opcount.util import get_work_for_function
import pandas as pd

@dataclass
class DataPoint:
    """Data point to graph."""

    input_size: int
    flops_per_cycle: float


def compute_ops_per_cycle(data_row: pd.Series) -> float:
    """Compute ops per cycle."""
    # print(f'input size = {env_data.input_size}, cycles = {data_row.cycles}')
    return get_work_for_function(data_row) / data_row.cycles
