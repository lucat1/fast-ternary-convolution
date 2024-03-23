#include "common.hpp"
#include "quantize.hpp"

// Quantize the input x to be {+1, 0, -1} 
// Input:
//   x: the data to be quantized, using N_C_H_W data format
//   padding1: the padding around Height
//   padding2: the padding around Width
//   ths: the threshold values of each filter or input image or activation
//   N: batch size or filter number, C: Channel, H: Height, W: Width
// Output:
//   qx: the quantized x, using N, H, W, C, B format
std::vector<int64_t> ternarize_NCHW_to_NHWCB(float* input, int padding_height,
         int padding_width, float* quant_threshold, int batch_size, int C,
         int input_height, int input_width) {
    const int64_t one = 1;
    int64_t onebit[CNTBITS];
    // 64-bits, set each bit
    for (int i = 0; i < CNTBITS; i++) {
        onebit[i] = one << i;
    }

    // initial packed channel num
    const int priChannel = C / CNTBITS;
    // packed_channels: actual packed input channel
    const int packed_channels = (C % CNTBITS) ? (priChannel + 1) : priChannel;
    const int packed_height = input_height + 2 * padding_height;
    const int packed_width = input_width + 2 * padding_width;
    // The quantized qx, in N_H_W_C_B format
    std::vector<int64_t> qx = std::vector<int64_t>(batch_size * packed_height * packed_width * packed_channels * BITS, 0);
    int64_t* qxptr = qx.data();

    for (int in = 0; in < batch_size; in++) {
        for (int ih = 0; ih < input_height; ih++) {
            for (int iw = 0; iw < input_width; iw++) {

                // Pack the first part: 0 ~ priChannel*CNTBITS
                for (int ic = 0; ic < priChannel; ic++) {
                    // for 2-bit packing
                    int64_t p1 = 0;
                    int64_t p2 = 0;
                    for (int bit = 0; bit < CNTBITS; bit++) {
                        // PyTorch uses N_C_H_W format
                        // x.index({in, ic*CNTBITS+bit, ih, iw})
                        float currentx = input[((in * C + (ic * CNTBITS + bit)) * input_height + ih) * input_width + iw];
                        if (currentx > quant_threshold[in]) {
                            // Pack 1: 01

                            p2 = p2 | onebit[bit];
                        }
                        else if (currentx < (-quant_threshold[in])) {
                            // Pack -1: 11
                            p1 = p1 | onebit[bit];
                            p2 = p2 | onebit[bit];
                        }
                    }
                    // Store the ternarized and packed data in N_H_W_C_B format
                    //qx.index({ in, ih + padding1, iw + padding2, priChannel * 2 + 0 }) = p1;
                    //qx.index({ in, ih + padding1, iw + padding2, priChannel * 2 + 1 }) = p2;
                    qxptr[(((in * packed_height + ih + padding_height) * packed_width + iw + padding_width) * packed_channels + ic) * BITS + 0] = p1;
                    qxptr[(((in * packed_height + ih + padding_height) * packed_width + iw + padding_width) * packed_channels + ic) * BITS + 1] = p2;
                }

                // Pack the second part: priChannel*CNTBITS ~ C
                if ((C % CNTBITS) > 0) {
                    int64_t p1 = 0;
                    int64_t p2 = 0;
                    for (int bit = 0; bit < (C % CNTBITS); bit++) {
                        float currentx = input[((in * C + (priChannel * CNTBITS + bit)) * input_height + ih) * input_width + iw];
                        if (currentx > quant_threshold[in]) {
                            // Pack 1: 01

                            p2 = p2 | onebit[bit];
                        }
                        else if (currentx < (-quant_threshold[in])) {
                            // Pack -1: 11
                            p1 = p1 | onebit[bit];
                            p2 = p2 | onebit[bit];
                        }
                    }
                    // Old NCHWB format for reference
                    //qxptr[((in * packed_channels + priChannel) * packed_height + (ih + padding1)) * packWB + ow + 0] = p1;
                    //qxptr[((in * packed_channels + priChannel) * packed_height + (ih + padding1)) * packWB + ow + 1] = p2;

                    // Store packed data into new NHWCB format
                    qxptr[(((in * packed_height + ih + padding_height) * packed_width + iw + padding_width) * packed_channels + priChannel) * BITS + 0] = p1;
                    qxptr[(((in * packed_height + ih + padding_height) * packed_width + iw + padding_width) * packed_channels + priChannel) * BITS + 1] = p2;
                }
            }
        }
    }
    return qx;
}


// Quantize the input x to be {+1, -1} 
// Input:
//   x: the data to be quantized, using N_C_H_W data format
//   padding1: the padding around Height
//   padding2: the padding around Width
//   ths: the threshold values of each filter or input image or activation. Default: 0
//   N: batch size or filter number, C: Channel, H: Height, W: Width
// Output:
//   qx: the quantized x, using N, H, W, C format
std::vector<int64_t> Binarize_NCHW_to_NHWC(const float* input, int padding_height, int padding_width, float* quant_threshold, int batch_size, int C, int input_height, int input_width) {
    const int64_t one = 1;
    int64_t onebit[CNTBITS];
    // 64-bits, set each bit
    for (int i = 0; i < CNTBITS; i++) {
        onebit[i] = one << i;
    }

    // initial packed channel num
    const int priChannel = C / CNTBITS;
    // packed_channels: actual packed input channel
    const int packed_channels = (C % CNTBITS) ? (priChannel + 1) : priChannel;
    const int packed_height = input_height + 2 * padding_height;
    const int packed_width = input_width + 2 * padding_width;

    // The PyTorch data always uses N, C, H, W format, no matter how we permute the data
    // torch::Tensor qx = torch::zeros({ N, packed_height, packed_width, packed_channels }, torch::dtype(torch::kInt64));
    std::vector<int64_t> qx = std::vector<int64_t>(batch_size * packed_height * packed_width * packed_channels, 0);
    int64_t* qxptr = qx.data();

    for (int in = 0; in < batch_size; in++) {
        for (int ih = 0; ih < input_height; ih++) {
            for (int iw = 0; iw < input_width; iw++) {

                // Pack the first part: 0 ~ priChannel*CNTBITS
                for (int ic = 0; ic < priChannel; ic++) {
                    // for 1-bit packing
                    int64_t p1 = 0;
                    for (int bit = 0; bit < CNTBITS; bit++) {
                        // PyTorch uses N_C_H_W format: x.index({in, ic*CNTBITS+bit, ih, iw})
                        // Each filter can have its own adjustable quantization threshold, e.g., -0.1, 0, +0.1, ...
                        if (input[((in * C + (ic * CNTBITS + bit)) * input_height + ih) * input_width + iw] < quant_threshold[in]) {
                            // Pack -1: 1
                            p1 = p1 | onebit[bit];
                        }
                    }
                    // Store the binarized and packed data in N_H_W_C format
                    qxptr[((in * packed_height + ih + padding_height) * packed_width + iw + padding_width) * packed_channels + ic] = p1;
                }

                // Pack the second part: priChannel*CNTBITS ~ C
                if ((C % CNTBITS) > 0) {
                    int64_t p1 = 0;
                    for (int bit = 0; bit < (C % CNTBITS); bit++) {
                        if (input[((in * C + (priChannel * CNTBITS + bit)) * input_height + ih) * input_width + iw] < quant_threshold[in]) {
                            // Pack -1: 1
                            p1 = p1 | onebit[bit];
                        }
                    }
                    qxptr[((in * packed_height + ih + padding_height) * packed_width + iw + padding_width) * packed_channels + priChannel] = p1;
                }
            }
        }
    }
    return qx;
}


// This Binarization use ths=0
std::vector<int64_t> binarize_NCHW_to_NHWC(const float* input, int padding_height, int padding_width, int batch_size, int C, int input_height, int input_width) {
    const int64_t one = 1;
    int64_t onebit[CNTBITS];
    // 64-bits, set each bit
    for (int i = 0; i < CNTBITS; i++) {
        onebit[i] = one << i;
    }

    // initial packed channel num
    const int priChannel = C / CNTBITS;
    // packed_channels: actual packed input channel
    const int packed_channels = (C % CNTBITS) ? (priChannel + 1) : priChannel;
    const int packed_height = input_height + 2 * padding_height;
    const int packed_width = input_width + 2 * padding_width;

    // The PyTorch data always uses N, C, H, W format, no matter how we permute the data
    // torch::Tensor qx = torch::zeros({ N, packed_height, packed_width, packed_channels }, torch::dtype(torch::kInt64));
    std::vector<int64_t> qx = std::vector<int64_t>(batch_size * packed_height * packed_width * packed_channels, 0);
    int64_t* qxptr = qx.data();

    for (int in = 0; in < batch_size; in++) {
        for (int ih = 0; ih < input_height; ih++) {
            for (int iw = 0; iw < input_width; iw++) {

                // Pack the first part: 0 ~ priChannel*CNTBITS
                for (int ic = 0; ic < priChannel; ic++) {
                    // for 1-bit packing
                    int64_t p1 = 0;
                    for (int bit = 0; bit < CNTBITS; bit++) {
                        // PyTorch uses N_C_H_W format: x.index({in, ic*CNTBITS+bit, ih, iw})
                        // Each channel can have its own adjustable quantization threshold, e.g., -0.1, 0, +0.1, ...
                        if (input[((in * C + (ic * CNTBITS + bit)) * input_height + ih) * input_width + iw] < 0) {
                            // Pack -1: 1
                            p1 = p1 | onebit[bit];
                        }
                    }
                    // Store the binarized and packed data in N_H_W_C format
                    qxptr[((in * packed_height + ih + padding_height) * packed_width + iw + padding_width) * packed_channels + ic] = p1;
                }

                // Pack the second part: priChannel*CNTBITS ~ C
                if ((C % CNTBITS) > 0) {
                    int64_t p1 = 0;
                    for (int bit = 0; bit < (C % CNTBITS); bit++) {
                        if (input[((in * C + (priChannel * CNTBITS + bit)) * input_height + ih) * input_width + iw] < 0) {
                            // Pack -1: 1
                            p1 = p1 | onebit[bit];
                        }
                    }
                    qxptr[((in * packed_height + ih + padding_height) * packed_width + iw + padding_width) * packed_channels + priChannel] = p1;
                }
            }
        }
    }
    return qx;
}


std::vector<int> btn_cnt_w2(int64_t* quantized_weights, int output_channels, int input_channels, int kernel_height, int kernel_width) {
    int num_packed_channels = (input_channels % CNTBITS) ? (input_channels / CNTBITS + 1) : (input_channels / CNTBITS);
    std::vector<int> y = std::vector<int>(KN, 0);

    for (int n = 0; n < output_channels; n++) {
        for (int h = 0; h < kernel_height; h++) {
            for (int w = 0; w < kernel_width; w++) {
                for (int c = 0; c < num_packed_channels; c++) {
                    y[n] += popcnt64(quantized_weights[(((n * kernel_height + h) * kernel_width + w) * num_packed_channels + c) * BITS + 1]);      
                }
            }
        }
    }

    return y;
}