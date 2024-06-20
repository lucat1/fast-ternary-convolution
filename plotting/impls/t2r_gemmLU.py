from plotting.impl import Cost, Impl
from plotting.utils import CNTBITS, BITS, POPCNT_OPS
from math import ceil
import pandas as pd
from plotting.impl.baseline import Baseline


class T2RGemmLU(Baseline):
    """Op count for t2r_gemmLU implementation."""

    def __init__(self, parameters: pd.Series):
        """Invoke Baseline for initialization."""
        super.__init__(parameters)

    @classmethod
    def t2r(self) -> Cost:
        """Get merged tern2row op count."""
        iops = 0
        flops = 0
        q = 0

        # onebit[i] = (int64_t)1 << i
        iops += CNTBITS
        q += 8 * CNTBITS

        
        
        raise NotImplementedError()

    @classmethod
    def gemmLU(self) -> Cost:
        """Get merged gemmLU op count."""
        raise NotImplementedError()
