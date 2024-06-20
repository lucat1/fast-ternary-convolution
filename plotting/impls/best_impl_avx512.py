from plotting.impl import Cost, Impl
from plotting.utils import CNTBITS, BITS, POPCNT_OPS, get_input_size
from math import ceil
import pandas as pd
from plotting.impl.t2r_gemmLU import T2RGemmLU

class BestImplAVX512(T2RGemmLU):
    """Op count for best_impl_avx2."""

    def __init__(self, parameters: pd.Series):
        """Invoke Baseline for initialization."""
        super.__init__(parameters)

    @classmethod
    def ternarize_im2row(self) -> Cost:
        """Get t2r_avx2 op count."""
        # Bytes transferred is the same as tern2row_cpy.
        q = super().ternarize_im2row().q

        iops = 0
        flops = 0

        # onebit[i] = one << i;
        iops += CNTBITS

        vec_loops = CNTBITS // 16

        # _mm512_cmp_ps_mask(current_values, curr_thresh_avx512, _CMP_GT_OS)
        flops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 16

        # _mm512_cmp_ps_mask(current_values, neg_curr_thresh_avx512, _CMP_LT_OS)
        flops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 16

        # __mmask16 current_values_gt_or_lt = current_values_gt_thresh | current_values_lt_neg_thresh;
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops

        # __mmask8 lower_first_bits_mask = (__mmask8)(current_values_lt_neg_thresh & 0xFF);
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops

        # __mmask8 lower_second_bits_mask = (__mmask8)(current_values_gt_or_lt & 0xFF);
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops

        # __mmask8 upper_first_bits_mask = (__mmask8)(current_values_lt_neg_thresh >> 8);
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops

        # __mmask8 upper_second_bits_mask = (__mmask8)(current_values_gt_or_lt >> 8);
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops

        # first_bits_lower = _mm512_mask_or_epi64(first_bits_lower, lower_first_bits_mask, first_bits_lower, one_bit_lower);
        iops += .5 * self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 8

        # first_bits_upper = _mm512_mask_or_epi64(first_bits_upper, upper_first_bits_mask, first_bits_upper, one_bit_upper);
        iops += .5 * self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 8

        # second_bits_lower = _mm512_mask_or_epi64(second_bits_lower, lower_second_bits_mask, second_bits_lower, one_bit_lower);
        iops += .5 * self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 8

        # second_bits_upper = _mm512_mask_or_epi64(second_bits_upper, upper_second_bits_mask, second_bits_upper, one_bit_upper);
        iops += .5 * self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 8

        # first_bits = _mm512_reduce_or_epi64(_mm512_or_epi64(first_bits_lower, first_bits_upper));
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * 8 * 7

        # second_bits = _mm512_reduce_or_epi64(_mm512_or_epi64(second_bits_lower, second_bits_upper));
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * 8 * 7

        # first_bits |= first_bits0 | first_bits1;
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * 2

        # second_bits |= second_bits0 | second_bits1;
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * 2

        iops = ceil(iops)
        flops = ceil(flops)

        return Cost(iops=iops, flops=flops, q=q)

    @classmethod
    def gemm_prelu(self) -> Cost:
        """Get merged gemmLU op count."""
        raise NotImplementedError()
