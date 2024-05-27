#include "impl/baseline_original/tab.hpp"
#include "common.hpp"
#include "impl/baseline_original/activation.hpp"
#include "impl/baseline_original/gemm.hpp"
#include "impl/baseline_original/im2row.hpp"
#include "impl/baseline_original/quantize.hpp"
#include "measure.hpp"

namespace baseline_original {
Tensor4D<float> conv(const Tensor4D<float> &_input,
                     const Tensor1D<float> &_thresholds,
                     const size_t _padding_h, const size_t _padding_w,
                     const Tensor5D<int64_t> &_kernel, const size_t _stride_h,
                     const size_t _stride_w, float _relu_alpha) {
  /* Container function of quantization and convolution functions. Can be
   * applied to conv and FC layers. Conv: 1X1, 3X3, and larger kernels. FC
   * equals to 1x1 conv. type: 0: TAB-TNN 1: TAB-TBN 2: TAB-BTN 3: TAB-BNN
   * */
  // Ternary and Binary Convolution using N, H, W, C, B format
  // Input:
  //   x: input activation in NCHW format
  //   qw: quantized weights in KN_KH_KW_C_Bit format
  //   stride: the stride on Height and Width
  //   padding: the padding on Height and Width
  //   N: batch number, C, channel, H: Height, W: Width
  //   KN: number of filters/kernels, KH: Kernel Height, KW, Kernel Width
  // Output:
  //   y: convolution result

  // Adapting our signature to the original one
  float *X = _input.data;
  float *Q_Threshold = _thresholds.data;
  int64_t *QWeights = _kernel.data;
  // NOTE We specialize to ternary.
  int *BTN_CNT1 = nullptr;
  ConvolutionType TYPE = ConvolutionType::TNN;
  int PaddingH = _padding_h;
  int PaddingW = _padding_w;
  int StrideH = _stride_h;
  int StrideW = _stride_w;
  int Batch_Size = _input.dim1;
  int C = _input.dim2;
  int H = _input.dim3;
  int W = _input.dim4;
  int KN = _kernel.dim1;
  int KH = _kernel.dim2;
  int KW = _kernel.dim3;
  float ReLU_alpha = _relu_alpha;

  // Start of the algorithm
  int PackedH, PackedW, OH, OW, PackedC;

  PackedH = H + 2 * PaddingH; // Height after bit-packing
  PackedW = W + 2 * PaddingW; // Width  after bit-packing
  // Referring to
  // https://pytorch.org/docs/2.3/generated/torch.nn.Conv2d.html#conv2d
  OH = (PackedH - KH) / StrideH + 1; // Output Height
  OW = (PackedW - KW) / StrideW + 1; // Output Width
  PackedC = (C % CNTBITS) ? ((C / CNTBITS) + 1)
                          : (C / CNTBITS); // The channel after bit-packing

  std::vector<int64_t> qx;
  std::vector<int> yi;

  // Quantize and Img2Row/Img2Col
  if ((TYPE == ConvolutionType::TNN) || (TYPE == ConvolutionType::TBN)) {
    measure_point(measurement_point::ternarize, MeasurementEvent::START);
    qx = Ternarize_NCHW_to_NHWCB(X, PaddingH, PaddingW, Q_Threshold, Batch_Size,
                                 C, H, W);
    measure_point(measurement_point::ternarize, MeasurementEvent::END);
    measure_point(measurement_point::im2row, MeasurementEvent::START);
    qx = Img2Row_NHWCB_to_N_OHOW_KHKWC(qx.data(), Batch_Size, PackedC * BITS,
                                       PackedH, PackedW, KH, KW, StrideH,
                                       StrideW);
    measure_point(measurement_point::im2row, MeasurementEvent::END);
  } else {
    qx = Binarize_NCHW_to_NHWC(X, PaddingH, PaddingW, Q_Threshold, Batch_Size,
                               C, H, W);
    qx = Img2Row_NHWCB_to_N_OHOW_KHKWC(qx.data(), Batch_Size, PackedC, PackedH,
                                       PackedW, KH, KW, StrideH, StrideW);
  }

  // Bitwise GEMM

  switch (TYPE) {
  case ConvolutionType::TNN: {
    yi = TNNGEMM_baseline(qx.data(), QWeights, Batch_Size * OH * OW, KN,
                          PackedC * KH * KW);
    break;
  }
  case ConvolutionType::TBN: {
    yi = TBNGEMM_baseline(qx.data(), QWeights, Batch_Size * OH * OW, KN,
                          PackedC * KH * KW);
    break;
  }
  case ConvolutionType::BTN: {
    yi = BTNGEMM_baseline(qx.data(), QWeights, BTN_CNT1, Batch_Size * OH * OW,
                          KN, PackedC * KH * KW);
    break;
  }
  case ConvolutionType::BNN: {
    yi = BNNGEMM_baseline(qx.data(), QWeights, Batch_Size * OH * OW, KN,
                          PackedC * KH * KW, C * KH * KW);
    break;
  }
  } // switch

  // Activation function: PReLU
  return PReLU(yi.data(), Batch_Size, KN, OH, OW, ReLU_alpha);
}
} // namespace baseline_original
