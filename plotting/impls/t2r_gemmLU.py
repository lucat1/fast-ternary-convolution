from plotting.impl import Cost, Impl
from plotting.utils import CNTBITS, BITS, POPCNT_OPS, get_input_size
from math import ceil
import pandas as pd
from plotting.impls.baseline import Baseline


class T2RGemmLU(Baseline):
    """Op count for t2r_gemmLU implementation."""

    def __init__(self, parameters: pd.Series):
        """Invoke Baseline for initialization."""
        super().__init__(parameters)

    # def copy_all_channels(self) -> int:
    #     """Get bytes transferred from copy_all_channels."""
    #     q = 0
    #     fbcpp = self.pri_channels + (1 if self.p.channels % CNTBITS else 0)

    #     # int64_t v0 = tensor7d_get(...)
    #     q += 8 * fbcpp

    #     # tensor7d_set
    #     q += 8 * fbcpp

    #     # int64_t v1 = tensor7d_get(...)
    #     q += 8 * fbcpp

    #     # tensor7d_set
    #     q += 8 * fbcpp

    #     return q

    def ternarize_im2row(self) -> Cost:
        """Get merged tern2row op count."""
        cost_ternarize = super().ternarize()
        # input
        q = 4 * get_input_size(self.p) # for the input tensor
        q += 4 * self.p.kernel_number # for threshold
        q += 8 * (2 * self.p.batch_size * self.output_height * self.output_width * self.p.kernel_height * self.p.kernel_width * self.packed_channels * 2) # for the output
        return Cost(cost_ternarize.iops, cost_ternarize.flops, q)

    def gemm_prelu(self) -> Cost:
        """Get merged gemmLU op count."""
        cost_gemm = super().gemm()
        cost_prelu = super().prelu()
        iops = cost_gemm.iops + cost_prelu.iops
        flops = cost_gemm.flops + cost_prelu.flops
        return Cost(iops, flops, cost_gemm.q)
