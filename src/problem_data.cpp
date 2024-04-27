#include "problem_data.hpp"
#include "problem_parameters.hpp"

Matrix2D::Matrix2D(size_t height, size_t width) : Size(height, width) {}

size_t Matrix2D::size() { return this->height * this->width; }

Matrix4D::Matrix4D(size_t fst_dim, size_t snd_dim, size_t height, size_t width)
    : Size(height, width) {
  this->fst_dim = fst_dim;
  this->snd_dim = snd_dim;
}

size_t Matrix4D::size() {
  return this->fst_dim * this->snd_dim * this->height * this->width;
}

Matrix4D Data::x_shape() {
  return Matrix4D(this->batch_size, this->num_channels, this->input_size.height,
                  this->input_size.width);
}

Matrix4D Data::y_shape() {
  uint32_t packed_height =
      this->input_size.height + 2 * this->padding_size.height;
  uint32_t packed_width = this->input_size.width + 2 * this->padding_size.width;
  uint32_t output_height =
      (packed_height - this->kernel_size.height + 1) / this->stride_size.height;
  uint32_t output_width =
      (packed_width - this->kernel_size.width + 1) / this->stride_size.width;

  // TODO: in this input this is num_channels. In the output, it is kernel
  // number. Does this make sense?
  return Matrix4D(this->batch_size, this->kernel_number, output_height,
                  output_width);
}

Data::Data(ConvolutionType conv_type, uint32_t batch_size,
           uint32_t num_channels, uint32_t kernel_number, Size input_size,
           Size kernel_size, Size padding_size, Size stride_size)
    : Parameters(conv_type, batch_size, num_channels, kernel_number, input_size,
                 kernel_size, padding_size, stride_size) {}
