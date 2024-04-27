#include "common.hpp"
#include <map>

std::map<ConvolutionType, std::string> __ctn = {{ConvolutionType::TNN, "TNN"},
                                                {ConvolutionType::TBN, "TBN"},
                                                {ConvolutionType::BTN, "BTN"},
                                                {ConvolutionType::BNN, "BNN"}};

std::string convolution_name(ConvolutionType t) { return __ctn[t]; }
