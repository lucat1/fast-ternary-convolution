#include "problem_data.hpp"
#include "common.hpp"

using namespace std;

Shape4D::Shape4D(size_t fst_dim, size_t snd_dim, size_t height, size_t width)
    : fst_dim(fst_dim), snd_dim(snd_dim), height(height), width(width),
      size(fst_dim * snd_dim * height * width) {}

Shape5D::Shape5D(size_t fst_dim, size_t snd_dim, size_t trd_dim, size_t height,
                 size_t width)
    : fst_dim(fst_dim), snd_dim(snd_dim), trd_dim(trd_dim), height(height),
      width(width), size(fst_dim * snd_dim * trd_dim * height * width) {}

Shape2D packed_size(Parameters &p, Shape2D base) {
  const int packed_height = base.height + 2 * p.padding_size.height;
  const int packed_width = base.width + 2 * p.padding_size.width;
  return Shape2D(packed_height, packed_width);
}

size_t pckd_chans(Parameters &p) {
  return (p.num_channels % CNTBITS) ? ((p.num_channels / CNTBITS) + 1)
                                    : (p.num_channels / CNTBITS);
}

Shape4D y_shape(Parameters &p) {
  auto packed_input_size = packed_size(p, p.input_size);
  uint32_t output_height =
      (packed_input_size.height - p.kernel_size.height) /
      p.stride_size.height + 1;
  uint32_t output_width =
      (packed_input_size.width - p.kernel_size.width) / p.stride_size.width + 1;

  // TODO: in this input this is num_channels. In the output, it is kernel
  // number. Does this make sense?
  return Shape4D(p.batch_size, p.kernel_number, output_height, output_width);
}

Data::Data(Parameters p)
    : Parameters(p), packed_input_size(packed_size(p, input_size)),
      packed_kernel_size(packed_size(p, kernel_size)),
      packed_channels(pckd_chans(p)),
      x(batch_size, num_channels, input_size.height, input_size.width, true),
      quant_threshold(max(kernel_number, batch_size), true),
      quant_weights(kernel_number, packed_kernel_size.height,
                    packed_kernel_size.width, packed_channels, BITS, true),
      // This only needs to be allocated when conv_type == BTN, but there it's
      // always allocated for convenience.
      btn_cnt(kernel_number, true), y(y_shape(p), true) {
  // TODO: ask for this
  // +1 on `x` is required as ternarize_* does an off-by-one access
  // This is required only when doing a BTN
}
