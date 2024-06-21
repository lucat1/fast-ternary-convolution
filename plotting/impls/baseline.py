from plotting.impl import Cost, Impl
from plotting.utils import CNTBITS, BITS, POPCNT_OPS
from math import ceil
import pandas as pd


class Baseline(Impl):
    def __init__(self, parameters: pd.Series) -> None:
        super().__init__(parameters)
        self.pri_channels = self.p.channels / CNTBITS
        self.rem_channels = self.p.channels % CNTBITS

        # packed input sizes after ternarize
        self.packed_height = self.p.input_height + 2 * self.p.padding_size
        self.packed_width = self.p.input_width + 2 * self.p.padding_size
        self.packed_channels = self.p.channels // 64 + (1 if self.p.channels % 64 > 0 else 0)

        self.output_height = (self.packed_height - self.p.kernel_height) // self.p.stride_size + 1
        self.output_width = (self.packed_width - self.p.kernel_width) // self.p.stride_size + 1

        # gemm takes (m x n) and (n x k), outputs (m k)
        self.m = self.p.batch_size * self.output_height * self.output_width
        self.n = self.p.kernel_number
        self.k = self.p.kernel_height * self.p.kernel_width * self.packed_channels

    def ternarize(self) -> Cost:
        iops = 0
        flops = 0
        q = 0

        # NOTE: probably we can ignore as it's a constant number of them
        # onebit[i] = one << i;
        iops += CNTBITS
        # onebit[i]
        q += 8 * CNTBITS
        # currentx > quant_threshold[in]
        flops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * CNTBITS
        # p2 = p2 | onebit[bit];
        iops += .5 * self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * CNTBITS
        # currentx < (-quant_threshold[in])
        flops += .5 * self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * CNTBITS
        # p1 = p1 | onebit[bit];
        # p2 = p2 | onebit[bit];
        iops += .25 * 2 * self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * CNTBITS
        # TODO: accout for assignments
        # qx[..] = p1
        # qx[..] = p2

        # currentx > quant_threshold[in]
        flops += self.p.batch_size * self.p.input_height * self.p.kernel_width * self.rem_channels
        # p2 = p2 | onebit[bit];
        iops += .5 * self.p.batch_size * self.p.input_height * self.p.input_width * self.rem_channels
        # currentx < (-quant_threshold[in])
        flops += .5 * self.p.batch_size * self.p.input_height * self.p.input_width * self.rem_channels
        # p1 = p1 | onebit[bit];
        # p2 = p2 | onebit[bit];
        iops += .25 * 2 * self.p.batch_size * self.p.input_height * self.p.input_width * self.rem_channels
        # TODO: accout for assignments
        # qx[..] = p1
        # qx[..] = p2

        iops = ceil(iops)
        flops = ceil(flops)

        # input[..], quant_threshold[..], onebit[..]
        q += (4+4+8) * (self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * CNTBITS)
        # 2x qx[..]
        q += (8+8) * (self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels)
        # input[..], quant_threshold[..]
        q += (4+4) * (self.p.batch_size * self.p.input_height * self.p.input_width * self.rem_channels)
        # 2x qx[..]
        q += (8+8) * (self.p.batch_size * self.p.input_height * self.p.input_width * self.rem_channels)

        return Cost(iops, flops, q)
    
    def im2row(self) -> Cost:
        iops = 0
        flops = 0

        out_h = (self.packed_height - self.p.kernel_height) // self.p.stride_size + 1;
        out_w = (self.packed_width - self.p.kernel_width) // self.p.stride_size + 1;

        count = self.p.batch_size * out_h * out_w * self.p.kernel_height * self.p.kernel_width * self.packed_channels * 2
        # one i64 is read and 1 is written (counts as 2) every iteration (count)
        q = 8*3*count

        return Cost(iops, flops, q)

    def gemm(self) -> Cost:
        k_bits = self.k*BITS

        iops = 0
        flops = 0

        # p1 = a[..] ^ b[..]
        iops += self.m * self.n * (k_bits/BITS)
        # p2 = a[..] & b[..]
        iops += self.m * self.n * (k_bits/BITS)
        # p1 & p2
        iops += self.m * self.n * (k_bits/BITS)
        # 2x popcnt
        iops += 2 * POPCNT_OPS * (self.m * self.n * (k_bits/BITS))
        # 2x +
        iops += 2 * (self.m * self.n * (k_bits/BITS))

        iops = ceil(iops)
        flops = ceil(flops)

        iter = self.m*self.n
        inner = iter*self.k
        q = 8*(4*inner + 2*iter)

        return Cost(iops, flops, q)

    def prelu(self) -> Cost:
        iops = 0
        flops = 0
        iter = self.p.batch_size * self.output_height * self.output_width * self.p.kernel_number

        # current > 0
        flops += iter
        # current * alpha
        flops += .5 * iter

        flops = ceil(flops)
        q = 4*3*iter

        return Cost(iops, flops, q)
