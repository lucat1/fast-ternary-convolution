"""Operation counts."""

import pandas as pd
from plot.opcount.conv import conv_op_count


def compute_ops(benchmark_info: pd.Series) -> int:
    """Get total number of operations."""
    return conv_op_count(benchmark_info)
