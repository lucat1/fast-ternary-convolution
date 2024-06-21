from plotting.impl import Cost, Impl
from plotting.utils import CNTBITS, BITS, POPCNT_OPS, get_input_size, M_BLOCK_SIZE, N_BLOCK_SIZE
from math import ceil
import pandas as pd
from plotting.impls.t2r_gemmLU import T2RGemmLU

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
        flops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 16

        # _mm512_cmp_ps_mask(current_values, neg_curr_thresh_avx512, _CMP_LT_OS)
        flops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 16

        # __mmask16 current_values_gt_or_lt = current_values_gt_thresh | current_values_lt_neg_thresh;
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops

        # __mmask8 lower_first_bits_mask = (__mmask8)(current_values_lt_neg_thresh & 0xFF);
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops

        # __mmask8 lower_second_bits_mask = (__mmask8)(current_values_gt_or_lt & 0xFF);
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops

        # __mmask8 upper_first_bits_mask = (__mmask8)(current_values_lt_neg_thresh >> 8);
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops

        # __mmask8 upper_second_bits_mask = (__mmask8)(current_values_gt_or_lt >> 8);
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops

        # first_bits_lower = _mm512_mask_or_epi64(first_bits_lower, lower_first_bits_mask, first_bits_lower, one_bit_lower);
        iops += .5 * self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 8

        # first_bits_upper = _mm512_mask_or_epi64(first_bits_upper, upper_first_bits_mask, first_bits_upper, one_bit_upper);
        iops += .5 * self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 8

        # second_bits_lower = _mm512_mask_or_epi64(second_bits_lower, lower_second_bits_mask, second_bits_lower, one_bit_lower);
        iops += .5 * self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 8

        # second_bits_upper = _mm512_mask_or_epi64(second_bits_upper, upper_second_bits_mask, second_bits_upper, one_bit_upper);
        iops += .5 * self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * vec_loops * 8

        # first_bits = _mm512_reduce_or_epi64(_mm512_or_epi64(first_bits_lower, first_bits_upper));
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * 8 * 7

        # second_bits = _mm512_reduce_or_epi64(_mm512_or_epi64(second_bits_lower, second_bits_upper));
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * 8 * 7

        # first_bits |= first_bits0 | first_bits1;
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * 2

        # second_bits |= second_bits0 | second_bits1;
        iops += self.p.batch_size * self.p.input_height * self.p.input_width * self.pri_channels * 2

        iops = ceil(iops)
        flops = ceil(flops)

        return Cost(iops=iops, flops=flops, q=q)

    @classmethod
    def gemm_kernel_512(self) -> Cost:
        """
        Get gemm_kernel_512 op count.

        TODO: is it ok to use Q from super class?
        """
        iops = 0
        flops = 0

        vec_loops = self.k // (8 * BITS)

        # TODO: Is this correct?
        rem_loops = (self.k % (8 * BITS)) // (4 * BITS)

        # TODO: Is this correct?
        last_loops = ((self.k % (8 * BITS)) % (4 * BITS)) // BITS

        # comp1 = (_mm512_xor_epi64(load3, load4));
        iops += vec_loops * 8

        # comp3 = (_mm512_xor_epi64(load5, load6));
        iops += vec_loops * 8

        # comp2 = (_mm512_and_epi64(load3, load4));
        iops += vec_loops * 8

        # comp4 = (_mm512_and_epi64(load5, load6));
        iops += vec_loops * 8

        # comp8 = (_mm512_popcnt_epi64(comp6));
        iops += vec_loops * 8 * POPCNT_OPS

        # comp7 = (_mm512_and_epi64(comp5, comp6));
        iops += vec_loops * 8

        # comp9 = (_mm512_popcnt_epi64(comp7));
        iops += vec_loops * 8 * POPCNT_OPS

        # comp10 = (_mm512_add_epi64(load1, comp8));
        iops += vec_loops * 8

        # comp11 = (_mm512_add_epi64(load2, comp9));
        iops += vec_loops * 8

        # comp14 = (((load9) ^ (load10)));
        iops += rem_loops

        # comp15 = (((load11) & (load12)));
        iops += rem_loops

        # comp16 = (popcnt64(comp15));
        iops += rem_loops * POPCNT_OPS

        # comp17 = (((comp14) & (comp15)));
        iops += rem_loops

        # comp18 = (popcnt64(comp17));
        iops += rem_loops * POPCNT_OPS

        # comp19 = (((load7) + (comp16)));
        iops += rem_loops

        # comp20 = (((load8) + (comp18)));
        iops += rem_loops

        # comp12 = (_mm512_reduce_add_epi64(load1));
        iops += 3

        # comp13 = (_mm512_reduce_add_epi64(load2));
        iops += 3

        # comp21 = (((comp12) + (load7)));
        iops += 1

        # comp22 = (((comp13) + (load8)));
        iops += 1

        # comp26 = (((load13) ^ (load14)));
        iops += last_loops

        # comp27 = (((load15) & (load16)));
        iops += last_loops

        # comp28 = (popcnt64(comp27));
        iops += last_loops * POPCNT_OPS

        # comp29 = (((comp26) & (comp27)));
        iops += last_loops

        # comp30 = (popcnt64(comp29));
        iops += last_loops * POPCNT_OPS

        # comp31 = (((comp21) + (comp28)));
        iops += last_loops

        # comp32 = (((comp22) + (comp30)));
        iops += last_loops

        # comp23 = (((comp21) - (comp22)));
        iops += 1

        # comp24 = (((comp23) - (comp22)));
        iops += 1

        # comp25 = (((((comp24) > (0))) ? (comp24) : (((comp24) * (alpha)))));
        flops += 1
        iops += .5

        return Cost(iops=iops, flops=flops, q=0)

    @classmethod
    def gemm_prelu(self) -> Cost:
        """
        Get merged gemmLU op count.

        TODO: is it ok to use Q from super class?
        """
        q = super().gemm_prelu
        cost_gemm_kernel_512 = self.gemm_kernel_512()

        cost = cost_gemm_kernel_512 * (self.m // M_BLOCK_SIZE * self.n // N_BLOCK_SIZE)
        cost += cost_gemm_kernel_512 * (self.m // M_BLOCK_SIZE * (self.n % N_BLOCK_SIZE))
        cost += cost_gemm_kernel_512 * ((self.m % M_BLOCK_SIZE) * (self.n // N_BLOCK_SIZE))
        cost += cost_gemm_kernel_512 * ((self.m % M_BLOCK_SIZE) * (self.n % N_BLOCK_SIZE))

        return Cost(iops=ceil(cost.iops), flops=ceil(cost.flops), q=q)
