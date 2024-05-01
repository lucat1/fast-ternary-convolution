#include "problem_data.hpp"
#include "alloc.hpp"
#include "common.hpp"
#include "problem_parameters.hpp"

using namespace std;

size_t Matrix2D::_size() { return height * width; }

Matrix2D::Matrix2D(size_t height, size_t width)
    : Size(height, width), size(_size()) {}

size_t Matrix4D::_size() { return fst_dim * snd_dim * height * width; }

Matrix4D::Matrix4D(size_t fst_dim, size_t snd_dim, size_t height, size_t width)
    : Size(height, width), fst_dim(fst_dim), snd_dim(snd_dim), size(_size()) {}

size_t Matrix5D::_size() {
  return fst_dim * snd_dim * trd_dim * height * width;
}

Matrix5D::Matrix5D(size_t fst_dim, size_t snd_dim, size_t trd_dim,
                   size_t height, size_t width)
    : Size(height, width), fst_dim(fst_dim), snd_dim(snd_dim), trd_dim(trd_dim),
      size(_size()) {}

Size Data::packed_size(Size base) {
  const int packed_height = base.height + 2 * padding_size.height;
  const int packed_width = base.width + 2 * padding_size.width;
  return Size(packed_height, packed_width);
}

Matrix4D Data::_x_shape() {
  return Matrix4D(batch_size, num_channels, input_size.height,
                  input_size.width);
}

size_t Data::_btn_cnt_size() { return kernel_number; }

size_t Data::_quant_threshold_size() { return max(kernel_number, batch_size); }

Matrix5D Data::_quant_weights_shape() {
  using namespace std;

  const int packed_channels = (num_channels % CNTBITS)
                                  ? ((num_channels / CNTBITS) + 1)
                                  : (num_channels / CNTBITS);

  return Matrix5D(kernel_number, packed_kernel_size.height,
                  packed_kernel_size.width, packed_channels, BITS);
}

Matrix4D Data::_y_shape() {
  uint32_t packed_height = input_size.height + 2 * padding_size.height;
  uint32_t packed_width = input_size.width + 2 * padding_size.width;
  uint32_t output_height =
      (packed_height - kernel_size.height + 1) / stride_size.height;
  uint32_t output_width =
      (packed_width - kernel_size.width + 1) / stride_size.width;

  // TODO: in this input this is num_channels. In the output, it is kernel
  // number. Does this make sense?
  return Matrix4D(batch_size, kernel_number, output_height, output_width);
}

Data::Data(ConvolutionType conv_type, uint32_t batch_size,
           uint32_t num_channels, uint32_t kernel_number, Size input_size,
           Size kernel_size, Size padding_size, Size stride_size,
           float relu_alpha)
    : Parameters(conv_type, batch_size, num_channels, kernel_number, input_size,
                 kernel_size, padding_size, stride_size),
      relu_alpha(relu_alpha), packed_input_size(packed_size(input_size)),
      packed_kernel_size(packed_size(kernel_size)), x_shape(_x_shape()),
      quant_threshold_size(_quant_threshold_size()),
      quant_weights_shape(_quant_weights_shape()),
      btn_cnt_size(_btn_cnt_size()), y_shape(_y_shape()) {
  // TODO: ask for this
  // The +1 is required as ternarize_* does an off-by-one access
  x = alloc::calloc<float>(x_shape.size + 1);
  quant_threshold = alloc::calloc<float>(quant_threshold_size);
  quant_weights = alloc::calloc<int64_t>(quant_weights_shape.size);
  // This is required only when doing a BTN
  btn_cnt = conv_type == ConvolutionType::BTN ? alloc::calloc<int>(btn_cnt_size)
                                              : nullptr;
  y = alloc::calloc<float>(y_shape.size);
}

Data::~Data() {
  if (x != nullptr)
    alloc::free(x);
  if (quant_threshold != nullptr)
    alloc::free(quant_threshold);

  if (quant_weights != nullptr)
    alloc::free(quant_weights);

  if (btn_cnt != nullptr)
    alloc::free(btn_cnt);

  if (y != nullptr)
    alloc::free(y);

  x = quant_threshold = y = nullptr;
  quant_weights = nullptr;
  btn_cnt = nullptr;
}
