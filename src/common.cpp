#include "common.hpp"
#include <map>

#include <fstream>
#include <vector>

map<ConvolutionType, string> __ctn = {{ConvolutionType::TNN, "TNN"},
                                      {ConvolutionType::TBN, "TBN"},
                                      {ConvolutionType::BTN, "BTN"},
                                      {ConvolutionType::BNN, "BNN"}};

string convolution_name(ConvolutionType t) { return __ctn[t]; }

InfraParameters::InfraParameters(uint32_t channels, size_t batch_size,
                                 size_t input_height, size_t input_width,
                                 uint32_t kernel_number, size_t kernel_height,
                                 size_t kernel_width, size_t padding_size,
                                 size_t stride_size)
    : channels(channels), batch_size(batch_size), input_height(input_height),
      input_width(input_width), kernel_number(kernel_number),
      kernel_height(kernel_height), kernel_width(kernel_width),
      padding_size(padding_size), stride_size(stride_size) {}

// From: https://en.cppreference.com/w/cpp/locale/ctype_char

// This ctype facet classifies commas and endlines as whitespace
struct csv_whitespace : ctype<char> {
  static const mask *make_table() {
    // make a copy of the "C" locale table
    static vector<mask> v(classic_table(), classic_table() + table_size);
    v[','] |= space;  // comma will be classified as whitespace
    v[' '] &= ~space; // space will not be classified as whitespace
    return &v[0];
  }

  csv_whitespace(::size_t refs = 0) : ctype(make_table(), false, refs) {}
};

fstream &operator>>(fstream &is, InfraParameters &params) {
  auto prev = is.imbue(std::locale(is.getloc(), new csv_whitespace));

  is >> params.channels;
  is >> params.batch_size;
  is >> params.input_height;
  is >> params.input_width;
  is >> params.kernel_number;
  is >> params.kernel_height;
  is >> params.kernel_width;
  is >> params.padding_size;
  is >> params.stride_size;

  is.imbue(prev);
  return is;
}
