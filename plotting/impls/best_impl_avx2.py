from plotting.impl import Cost, Impl
from plotting.utils import CNTBITS, BITS, POPCNT_OPS, get_input_size
from math import ceil
import pandas as pd
from plotting.impl.t2r_gemmLU import T2RGemmLU

class BestImplAVX2(T2RGemmLU):
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

        # _mm256_cmp_ps(current_values01, curr_thresh_avx, _CMP_GT_OS));
        flops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 4

        # _mm256_cmp_ps(current_values01, neg_curr_thresh_avx, _CMP_LT_OS)
        flops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 4

        # _mm256_or_si256(current_values01_gt_thresh, current_values01_lt_neg_thresh);
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 4

        # _mm256_cmp_ps(current_values23, curr_thresh_avx, _CMP_GT_OS));
        flops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 4

        # _mm256_cmp_ps(current_values23, neg_curr_thresh_avx, _CMP_LT_OS)
        flops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 4

        # _mm256_or_si256(current_values23_gt_thresh, current_values23_lt_neg_thresh);
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 4

        # TODO: I am not sure how many ops _mm256_cvtepi32_epi64 is...?
        # I will assume a sign-extending a single integer is one iop.
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 4

        # _mm256_or_si256(first_bits0, one_bits0)
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 4
        # _mm256_or_si256(first_bits1, one_bits1)
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 4
        # _mm256_or_si256(first_bits2, one_bits2)
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 4
        # _mm256_or_si256(first_bits3, one_bits3)
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 4

        # TODO: I am not sure how many ops _mm256_cvtepi32_epi64 is...?
        # I will assume a sign-extending a single integer is one iop.
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 4

        # _mm256_or_si256(second_bits0, one_bits0)
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 4
        # _mm256_or_si256(second_bits1, one_bits1)
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 4
        # _mm256_or_si256(second_bits2, one_bits2)
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 4
        # _mm256_or_si256(second_bits3, one_bits3)
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * vec_loops * 4
        
        # _mm256_or_si256(first_bits0, _mm256_or_si256(first_bits1, _mm256_or_si256(first_bits2, first_bits3)));
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * 3 * 4

        # _mm256_or_si256(first_bits_ored, first_bits_ored_perm)
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * 4

        # _mm256_or_si256(second_bits0, _mm256_or_si256(second_bits1, _mm256_or_si256(second_bits2, second_bits3)));
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * 3 * 4

        # _mm256_or_si256(second_bits_ored, second_bits_ored_perm)
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * 4

        # first_bits |= first_bits_0 | first_bits_1
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * 2

        # second_bits |= second_bits_0 | second_bits_1
        iops += self.p.batch_size * self.p.kernel_height * self.p.kernel_width * self.pri_channels * 2

        iops = ceil(iops)
        flops = ceil(flops)

        return Cost(iops=iops, flops=flops, q=q)

    @classmethod
    def gemm_prelu(self) -> Cost:
        """Get merged gemmLU op count."""
        if self.p.channels < 512:
            return super().gemm_prelu()
        raise NotImplementedError()
