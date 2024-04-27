#pragma once
#include "common.hpp"
#include <random>
#include <thread>

// direct zero padding function
template <typename T>
std::vector<T> DirectPad(T *x, int padding1, int padding2, int N, int C, int H,
                         int W) {
  const int packH = H + 2 * padding1;
  const int packW = W + 2 * padding2;

  // The PyTorch data always uses N, C, H, W format, no matter how we permute
  // the data torch::Tensor qx = torch::zeros({ N, packH, packW, packC },
  // torch::dtype(torch::kInt64));
  std::vector<T> qx = std::vector<T>(N * C * packH * packW, 0);
  T *qxptr = qx.data();

  for (int in = 0; in < N; in++) {
    for (int ic = 0; ic < C; ic++) {
      for (int ih = 0; ih < H; ih++) {
        for (int iw = 0; iw < W; iw++) {
          qxptr[((in * C + ic) * packH + (ih + padding1)) * packW + iw +
                padding2] = x[((in * C + ic) * H + ih) * W + iw];
        }
      }
    }
  }

  return qx;
}

// direct conv2d implemented in fp
std::vector<float> DirectConv2d_FP32(float *x, float *w, int stride1,
                                     int stride2, int N, int C, int H, int W,
                                     int KN, int KH, int KW) {
  const int OH = H - KH + 1;
  const int OW = W - KW + 1;
  const int FH = (int)(OH / stride1);
  const int FW = (int)(OW / stride2);

  std::vector<float> y = std::vector<float>(N * KN * FH * FW);
  float *yptr = y.data();

  for (int on = 0; on < N; on++) {
    for (int kn = 0; kn < KN; kn++) {
      for (int oh = 0; oh < FH; oh++) {
        for (int ow = 0; ow < FW; ow++) {
          float sum = 0;
          // kc = kc + 1
          for (int kc = 0; kc < C; kc++) {
            for (int kh = 0; kh < KH; kh++) {
              for (int kw = 0; kw < KW; kw++) {
                // Use N_C_H_W format
                sum += x[((on * C + kc) * H + (oh * stride1 + kh)) * W +
                         ow * stride2 + kw] *
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

// Generate a random binary or ternary array for testing
std::vector<float> generate_array(int size, bool Ternary) {
  std::vector<float> y = std::vector<float>(size, 0);
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(-1, 1);
  if (Ternary) {
    for (int i = 0; i < size; i++) {
      y[i] = distribution(generator);
    }
  } else {
    for (int i = 0; i < size; i++) {
      int current = distribution(generator);
      if (current == 0)
        y[i] = 1;
      else
        y[i] = current;
    }
  }
  return y;
}

// Direct Compare Implemented in FP
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

// Direct Compare Implemented in FP
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

// TODO: use this for debugging errors
template <typename T> void print_vec(std::string name, T *v, int size) {
  std::cout << "vec \"" << name << "\" size " << size << std::endl;
  int i;
  for (i = 0; i < size - 7; i += 8) {
    std::cout << i << ": ";
    for (int j = i; j < i + 8; ++j)
      std::cout << v[j] << " ";
    std::cout << std::endl;
  }
  if (i < size) {
    std::cout << i << ": ";
    for (; i < size; ++i)
      std::cout << v[i] << " ";
    std::cout << std::endl;
  }
}
