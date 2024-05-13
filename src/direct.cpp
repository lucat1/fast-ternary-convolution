#include "direct.hpp"

#include <vector>

// @daniel: I wrote a function to reshape from nchw to nhwc. This is pretty
// much what was in the conv function before. The usage is different:
// We only keep one version of the direct convolution and padding functions,
// and we change the input and kernel data before using it.

Tensor4D<float> reshape_nchw_nhwc(const Tensor4D<float> &src) {
  const size_t N = src.dim1;
  const size_t C = src.dim2;
  const size_t H = src.dim3;
  const size_t W = src.dim4;

  Tensor4D<float> dest = Tensor4D<float>(N, H, W, C, false);

  for (size_t in = 0; in < N; in++) {
    for (size_t ic = 0; ic < C; ic++) {
      for (size_t ih = 0; ih < H; ih++) {
        for (size_t iw = 0; iw < W; iw++) {
          float val = src.get(in, ic, ih, iw);
          dest.set(val, in, ic, ih, iw);
        }
      }
    }
  }

  return dest;
}

// Pads the given 4-dimensional tensor over the H and W dimensions
// Input: Tensor4D(N, H, W, C)
// Output: Tensor4D(N, H+2padding_h, 2+2padding_w, C)
Tensor4D<float> direct_pad(const Tensor4D<float> &x, const size_t padding_h,
                           const size_t padding_w) {
  const size_t N = x.dim1;
  const size_t H = x.dim2;
  const size_t W = x.dim3;
  const size_t C = x.dim4;
  const size_t packH = H + 2 * padding_h;
  const size_t packW = W + 2 * padding_w;

  Tensor4D<float> output = Tensor4D<float>(N, packH, packW, C, true);

  for (size_t in = 0; in < N; in++) {
    for (size_t ih = 0; ih < H; ih++) {
      for (size_t iw = 0; iw < W; iw++) {
        for (size_t ic = 0; ic < C; ic++) {
          float val = x.get(in, ih, iw, ic);
          output.set(val, in, ih + padding_h, iw + padding_w, ic);
        }
      }
    }
  }

  return output;
}

// Direct convolution
// Input:
//   - input: Tensor4D(N, H, W, C)
//   - kernel: Tensor4D(KN, KH, KW, C)
// Output: Tensor4D(N, H+2padding_h, 2+2padding_w, C)
Tensor4D<float> direct_conv(const Tensor4D<float> &input,
                            const Tensor4D<float> &kernel,
                            const size_t stride_height,
                            const size_t stride_width) {
  const size_t KN = kernel.dim1;
  const size_t N = input.dim1;
  const size_t KH = kernel.dim2;
  const size_t H = input.dim2;
  assert(KH <= H);
  const size_t KW = kernel.dim3;
  const size_t W = input.dim3;
  assert(KW <= W);

  assert(kernel.dim4 == input.dim4);
  const size_t C = kernel.dim4;

  const size_t OH = (H - KH) / stride_height + 1;
  const size_t OW = (W - KW) / stride_width + 1;

  Tensor4D<float> y = Tensor4D<float>(N, OH, OW, KN, false);

  for (size_t on = 0; on < N; on++) {
    for (size_t oh = 0; oh < OH; oh++) {
      for (size_t ow = 0; ow < OW; ow++) {
        for (size_t kn = 0; kn < KN; kn++) {
          float sum = 0;
          // kc = kc + 1
          for (size_t kh = 0; kh < KH; kh++) {
            for (size_t kw = 0; kw < KW; kw++) {
              for (size_t c = 0; c < C; c++) {
                // old: N_C_H_W format
                // sum += x[((on * C + kc) * H + (oh * stride_height + kh)) * W
                // +
                //          ow * stride_width + kw] *
                //        w[((kn * C + kc) * KH + kh) * KW + kw];
                float xx = input.get(on, oh * stride_height + kh,
                                     ow * stride_width + kw, c);
                float kk = kernel.get(kn, kh, kw, c);
                sum += xx * kk;
              }
            }
          }
          // yptr[((on * FH + oh) * FW + ow) * KN + kn] = sum;
          y.set(sum, on, oh, ow, kn);
        }
      }
    }
  }

  return y;
}
