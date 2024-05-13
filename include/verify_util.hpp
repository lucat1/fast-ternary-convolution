#pragma once

#include "common.hpp"
#include "tensor.hpp"
#include <iostream>
#include <vector>

std::vector<float> DirectConv2d_FP32NCHW(float *x, float *w,
                                         size_t stride_height,
                                         size_t stride_width, size_t N,
                                         size_t C, size_t H, size_t W,
                                         size_t KN, size_t KH, size_t KW) {
  assert(H >= KH);
  assert(W >= KW);
  const int OH = H - KH;
  const int OW = W - KW;
  const int FH = (int)(OH / stride_height) + 1;
  const int FW = (int)(OW / stride_width) + 1;

  std::vector<float> y = std::vector<float>(N * KN * FH * FW);
  float *yptr = y.data();

  for (size_t on = 0; on < N; on++) {
    for (size_t kn = 0; kn < KN; kn++) {
      for (int oh = 0; oh < FH; oh++) {
        for (int ow = 0; ow < FW; ow++) {
          float sum = 0;
          // kc = kc + 1
          for (size_t kc = 0; kc < C; kc++) {
            for (size_t kh = 0; kh < KH; kh++) {
              for (size_t kw = 0; kw < KW; kw++) {
                // Use N_C_H_W format
                sum += x[((on * C + kc) * H + (oh * stride_height + kh)) * W +
                         ow * stride_width + kw] *
                       w[((kn * C + kc) * KH + kh) * KW + kw];
              }
            }
          }
          yptr[((on * FH + oh) * FW + ow) * KN + kn] = sum;
        }
      }
    }
  }

  return y;
}

template <typename T>
int Compare_Tensor_NHWC(T *X, T *X2, int N, int C, int H, int W) {
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          // Use N_C_H_W format
          T xx = X[((n * H + h) * W + w) * C + c] -
                 X2[((n * H + h) * W + w) * C + c];
          // std::cout<<X[((n * C + c) * H + h) * W + w] <<" X2: "<< X2[((n * C
          // + c) * H + h) * W + w]<<std::endl;
          if ((xx > 0.01) || (xx < -0.01)) {
            std::cout << "n: " << n << ", h: " << h << ", w: " << w
                      << ", c: " << c;
            std::cout << ", X1: " << X[((n * H + h) * W + w) * C + c]
                      << ", X2: " << X2[((n * H + h) * W + w) * C + c]
                      << std::endl;
            return -1;
          }
        }
      }
    }
  }
  return 1;
}

template <typename T>
size_t Compare_Tensor_NHWC(T *X, T *X2, size_t N, size_t C, size_t H,
                           size_t W) {
  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < C; c++) {
      for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
          // Use N_C_H_W format
          T xx = X[((n * H + h) * W + w) * C + c] -
                 X2[((n * H + h) * W + w) * C + c];
          // std::cout<<X[((n * C + c) * H + h) * W + w] <<" X2: "<< X2[((n * C
          // + c) * H + h) * W + w]<<std::endl;
          if ((xx > 0.01) || (xx < -0.01)) {
            std::cout << "n: " << n << ", h: " << h << ", w: " << w
                      << ", c: " << c;
            std::cout << ", X1: " << X[((n * H + h) * W + w) * C + c]
                      << ", X2: " << X2[((n * H + h) * W + w) * C + c]
                      << std::endl;
            return -1;
          }
        }
      }
    }
  }
  return 1;
}

template <typename T>
int Compare_Tensor_BNN_Padding(T *X, T *X2, int N, int C, int H, int W,
                               int PaddingH, int PaddingW) {
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      for (int h = PaddingH; h < H - 2 * PaddingH; h++) {
        for (int w = PaddingH; w < W - 2 * PaddingW; w++) {
          // Use N_C_H_W format
          T xx = X[((n * H + h) * W + w) * C + c] -
                 X2[((n * H + h) * W + w) * C + c];
          // std::cout<<X[((n * C + c) * H + h) * W + w] <<" X2: "<< X2[((n * C
          // + c) * H + h) * W + w]<<std::endl;
          if ((xx > 0.01) || (xx < -0.01)) {
            std::cout << "n: " << n << ", h: " << h << ", w: " << w
                      << ", c: " << c;
            std::cout << ", X1: " << X[((n * H + h) * W + w) * C + c]
                      << ", X2: " << X2[((n * H + h) * W + w) * C + c]
                      << std::endl;
            return -1;
          }
        }
      }
    }
  }
  return 1;
}
