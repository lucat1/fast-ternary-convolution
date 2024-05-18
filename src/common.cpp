#include "common.hpp"
#include <map>

#include <fstream>
#include <sstream>
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

fstream &operator>>(fstream &is, InfraParameters &params) {
  std::string line;
  if (std::getline(is, line)) {
    std::stringstream ss(line);
    std::string item;

    if (std::getline(ss, item, ',')) {
      params.channels = std::stoi(item);
    }
    if (std::getline(ss, item, ',')) {
      params.batch_size = std::stoi(item);
    }
    if (std::getline(ss, item, ',')) {
      params.input_height = std::stoi(item);
    }
    if (std::getline(ss, item, ',')) {
      params.input_width = std::stoi(item);
    }
    if (std::getline(ss, item, ',')) {
      params.kernel_number = std::stoi(item);
    }
    if (std::getline(ss, item, ',')) {
      params.kernel_height = std::stoi(item);
    }
    if (std::getline(ss, item, ',')) {
      params.kernel_width = std::stoi(item);
    }
    if (std::getline(ss, item, ',')) {
      params.padding_size = std::stoi(item);
    }
    if (std::getline(ss, item, ',')) {
      params.stride_size = std::stoi(item);
    }
  }
  return is;
}
