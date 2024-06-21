from plotting.impl import Cost, Impl
from plotting.utils import CNTBITS, BITS, POPCNT_OPS, get_input_size, M_BLOCK_SIZE, N_BLOCK_SIZE
from math import ceil
import pandas as pd
from plotting.impls.t2r_gemmLU import T2RGemmLU


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
        flops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 8

        # _mm256_cmp_ps(current_values01, neg_curr_thresh_avx, _CMP_LT_OS)
        flops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 8

        # _mm256_or_si256(current_values01_gt_thresh, current_values01_lt_neg_thresh);
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 8

        # _mm256_cmp_ps(current_values23, curr_thresh_avx, _CMP_GT_OS));
        flops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 8

        # _mm256_cmp_ps(current_values23, neg_curr_thresh_avx, _CMP_LT_OS)
        flops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 8

        # _mm256_or_si256(current_values23_gt_thresh, current_values23_lt_neg_thresh);
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 8

        # TODO: I am not sure how many ops _mm256_cvtepi32_epi64 is...?
        # I will assume a sign-extending a single integer is one iop.
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 4

        # _mm256_or_si256(first_bits0, one_bits0)
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 4
        # _mm256_or_si256(first_bits1, one_bits1)
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 4
        # _mm256_or_si256(first_bits2, one_bits2)
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 4
        # _mm256_or_si256(first_bits3, one_bits3)
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 4

        # TODO: I am not sure how many ops _mm256_cvtepi32_epi64 is...?
        # I will assume a sign-extending a single integer is one iop.
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 4

        # _mm256_or_si256(second_bits0, one_bits0)
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 4
        # _mm256_or_si256(second_bits1, one_bits1)
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 4
        # _mm256_or_si256(second_bits2, one_bits2)
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 4
        # _mm256_or_si256(second_bits3, one_bits3)
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 4

        # _mm256_or_si256(first_bits0, _mm256_or_si256(first_bits1, _mm256_or_si256(first_bits2, first_bits3)));
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * 3 * 4

        # _mm256_or_si256(first_bits_ored, first_bits_ored_perm)
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * 4

        # _mm256_or_si256(second_bits0, _mm256_or_si256(second_bits1, _mm256_or_si256(second_bits2, second_bits3)));
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * 3 * 4

        # _mm256_or_si256(second_bits_ored, second_bits_ored_perm)
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * 4

        # first_bits |= first_bits_0 | first_bits_1
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * 2

        # second_bits |= second_bits_0 | second_bits_1
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * 2

        iops = ceil(iops)
        flops = ceil(flops)

        return Cost(iops=iops, flops=flops, q=q)

    @classmethod
    def gemm_kernel_256(self) -> Cost:
        """
        Get gemm_kernel_256 op count.

        TODO: is it ok to use Q from super class?
        """
        iops = 0
        flops = 0

        vec_loops = self.k // (2 * 4 * BITS)

        # TODO: Is this correct?
        rem_loops = (self.k % (2 * 4 * BITS)) // (4 * BITS)

        # TODO: is this correct?
        last_loops = ((self.k % (2 * 4 * BITS)) % (4 * BITS)) // BITS

        # comp1 = (_mm256_xor_si256(load1, load2));
        iops += vec_loops * 4

        # comp3 = (_mm256_xor_si256(load3, load4));
        iops += vec_loops * 4

        # comp2 = (_mm256_and_si256(load1, load2));
        iops += vec_loops * 4

        # comp4 = (_mm256_and_si256(load3, load4));
        iops += vec_loops * 4

        # comp7 = (_mm256_and_si256(comp5, comp6));
        iops += vec_loops * 4

        # comp8 = (_mm256_xor_si256(load5, load6));
        iops += vec_loops * 4

        # comp10 = (_mm256_xor_si256(load7, load8));
        iops += vec_loops * 4

        # comp9 = (_mm256_and_si256(load5, load6));
        iops += vec_loops * 4

        # comp11 = (_mm256_and_si256(load7, load8));
        iops += vec_loops * 4

        # comp43 = (popcnt(declr1, sizeof(declr1)));
        iops += POPCNT_OPS

        # comp44 = (popcnt(declr2, sizeof(declr2)));
        iops += POPCNT_OPS

        # comp15 = (((load11) ^ (load12)));
        iops += rem_loops

        # comp16 = (((load13) & (load14)));
        iops += rem_loops

        # comp17 = (popcnt64(comp16));
        iops += rem_loops * POPCNT_OPS

        # comp18 = (((comp15) & (comp16)));
        iops += rem_loops

        # comp19 = (popcnt64(comp18));
        iops += rem_loops * POPCNT_OPS

        # comp20 = (((load9) + (comp17)));
        iops += rem_loops

        # comp21 = (((load10) + (comp19)));
        iops += rem_loops

        # comp22 = (((load17) ^ (load18)));
        iops += rem_loops

        # comp23 = (((load19) & (load20)));
        iops += rem_loops

        # comp24 = (popcnt64(comp23));
        iops += rem_loops * POPCNT_OPS

        # comp25 = (((comp22) & (comp23)));
        iops += rem_loops

        # comp26 = (popcnt64(comp25));
        iops += rem_loops * POPCNT_OPS

        # comp27 = (((load15) + (comp24)));
        iops += rem_loops

        # comp28 = (((load16) + (comp26)));
        iops += rem_loops

        # comp29 = (((load23) ^ (load24)));
        iops += rem_loops

        # comp30 = (((load25) & (load26)));
        iops += rem_loops

        # comp31 = (popcnt64(comp30));
        iops += rem_loops * POPCNT_OPS

        # comp32 = (((comp29) & (comp30)));
        iops += rem_loops

        # comp33 = (popcnt64(comp32));
        iops += rem_loops * POPCNT_OPS

        # comp34 = (((load21) + (comp31)));
        iops += rem_loops

        # comp35 = (((load22) + (comp33)));
        iops += rem_loops

        # comp36 = (((load29) ^ (load30)));
        iops += rem_loops

        # comp37 = (((load31) & (load32)));
        iops += rem_loops

        # comp38 = (popcnt64(comp37));
        iops += rem_loops * POPCNT_OPS

        # comp39 = (((comp36) & (comp37)));
        iops += rem_loops

        # comp40 = (popcnt64(comp39));
        iops += rem_loops * POPCNT_OPS

        # comp41 = (((load27) + (comp38)));
        iops += rem_loops

        # comp42 = (((load28) + (comp40)));
        iops += rem_loops

        # comp56 = (((load33) ^ (load34)));
        iops += last_loops

        # comp57 = (((load35) & (load36)));
        iops += last_loops

        # comp58 = (popcnt64(comp57));
        iops += last_loops * POPCNT_OPS

        # comp59 = (((comp56) & (comp57)));
        iops += last_loops

        # comp60 = (popcnt64(comp59));
        iops += last_loops * POPCNT_OPS

        # comp61 = (((comp48) + (comp58)));
        iops += last_loops

        # comp62 = (((comp52) + (comp60)));
        iops += last_loops

        # comp53 = (((comp48) - (comp52)));
        iops += 1

        # comp54 = (((comp53) - (comp52)));
        iops += 1

        # comp55 = (((((comp54) > (0))) ? (comp54) : (((comp54) * (alpha)))));
        flops += 1
        iops += .5

        # q already computed prior I believe.
        return Cost(iops=iops, flops=flops, q=0)

    @classmethod
    def gemm_prelu(self) -> Cost:
        """
        Get merged gemmLU op count.

        TODO: is it ok to use Q from super class?
        """
        super_gemm = super().gemm_prelu()

        if self.p.channels < 512:
            return super_gemm

        # TODO: bytes transferred should be the same right?
        q = super_gemm.q
        del super_gemm

        cost_gemm_kernel_256 = self.gemm_kernel_256()

        cost = cost_gemm_kernel_256 * (self.m // M_BLOCK_SIZE * self.n // N_BLOCK_SIZE)
        cost += cost_gemm_kernel_256 * (self.m // M_BLOCK_SIZE * (self.n % N_BLOCK_SIZE))
        cost += cost_gemm_kernel_256 * ((self.m % M_BLOCK_SIZE) * (self.n // N_BLOCK_SIZE))
        cost += cost_gemm_kernel_256 * ((self.m % M_BLOCK_SIZE) * (self.n % N_BLOCK_SIZE))

        return Cost(iops=ceil(cost.iops), flops=ceil(cost.flops), q=q)
