#include "bench.hpp"
#include "measure.hpp"
#include "problem_data.hpp"

#include <iomanip>
#include <iostream>
#include <map>
#include <random>

// uncomment this for a detailed (visual) performance measurement
#define PRINT_ALL

using namespace std;

std::vector<InfraParameters> bench_cases = {
    {64, 56, 56, 64, 3, 3, 1, 1},    {64, 56, 56, 128, 3, 3, 1, 1},
    {128, 28, 28, 128, 3, 3, 1, 1},  {128, 28, 28, 256, 3, 3, 1, 1},
    {256, 14, 14, 256, 3, 3, 1, 1},  {256, 14, 14, 512, 3, 3, 1, 1},

    {80, 224, 224, 80, 3, 3, 1, 1},  {80, 224, 224, 80, 3, 3, 1, 2},
    {80, 224, 224, 80, 3, 3, 1, 3},  {80, 224, 224, 80, 3, 3, 1, 4},

    {512, 56, 56, 256, 1, 1, 0, 1},  {512, 56, 56, 256, 3, 3, 1, 1},
    {512, 56, 56, 256, 5, 5, 2, 1},  {512, 56, 56, 256, 7, 7, 3, 1},
    {512, 56, 56, 256, 9, 9, 3, 1},  {512, 56, 56, 256, 11, 11, 3, 1},

    {2000, 1, 1, 4000, 1, 1, 0, 1},  {4000, 1, 1, 8000, 1, 1, 0, 1},
    {8000, 1, 1, 16000, 1, 1, 0, 1}, {16000, 1, 1, 32000, 1, 1, 0, 1},
};

class BenchData : public Data {
private:
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution;

  void randomize(float *dst, size_t size, bool ternary) {
    if (ternary)
      for (size_t i = 0; i < size; ++i)
        dst[i] = distribution(generator);
    else
      for (size_t i = 0; i < size; ++i) {
        auto v = distribution(generator);
        dst[i] = v == 0 ? 1 : v;
      }
  }

public:
  BenchData(ConvolutionType conv_type, uint32_t batch_size,
            uint32_t num_channels, uint32_t kernel_number, Size input_size,
            Size kernel_size, Size padding_size, Size stride_size,
            float relu_alpha)
      : Data(conv_type, batch_size, num_channels, kernel_number, input_size,
             kernel_size, padding_size, stride_size, relu_alpha) {
    distribution = std::uniform_int_distribution<int>(-1, 1);

    // TODO: ask for this
    // The +1 is required as ternarize_* does an off-by-one access
    randomize(x, x_shape.size + 1, has_ternary_input(conv_type));

    for (size_t i = 0; i < quant_threshold_size; ++i)
      quant_threshold[i] = 0.5;
  }
};

vector<vector<Interval>> one_run(Implementation impl, Data &data) {
  size_t num_runs = 30;
  vector<vector<Interval>> measurement_intervals;
  auto m = Measure::get_instance();

  for (size_t i = 0; i < num_runs; ++i) {
    m->reset();

    m->track(MeasurementFunction::CONV, MeasurementEvent::START);
    impl.fn(data.conv_type, data.btn_cnt, data.x, data.input_size.height,
            data.input_size.width, data.padding_size.height,
            data.padding_size.width, data.quant_threshold, data.num_channels,
            data.quant_weights, data.batch_size, data.stride_size.height,
            data.stride_size.height, data.kernel_number,
            data.kernel_size.height, data.kernel_size.width, data.relu_alpha,
            data.y);
    m->track(MeasurementFunction::CONV, MeasurementEvent::END);

    measurement_intervals.push_back(m->intervals());
  }

  return measurement_intervals;
}

map<MeasurementFunction, uint64_t> average(vector<vector<Interval>> raw) {
  map<MeasurementFunction, vector<Interval>> by_func;
  for (auto one_run : raw) {
    for (auto interval : one_run) {
      if (!by_func.contains(interval.func))
        by_func.insert({interval.func, {}});
      by_func[interval.func].push_back(interval);
    }
  }

  map<MeasurementFunction, uint64_t> avgs;
  for (auto event : by_func) {
    auto func = event.first;
    uint64_t tot = 0;

    for (auto interval : event.second)
      tot += interval.end_time - interval.start_time;

    uint64_t avg = tot / event.second.size();
    avgs.insert({func, avg});
  }

  return avgs;
}

const constexpr size_t conv_type_space = 3;
const constexpr size_t time_space = 12;
const constexpr size_t num_channels_space = 5;
const constexpr size_t batch_size_space = 2;
const constexpr size_t kernel_number_space = 4;
const constexpr size_t input_height_space = 3;
const constexpr size_t input_width_space = 3;
const constexpr size_t kernel_height_space = 2;
const constexpr size_t kernel_width_space = 2;
const constexpr size_t padding_size_space = 1;
const constexpr size_t stride_size_space = 1;

void print_line(string impl_name, ConvolutionType ct, MeasurementFunction mf,
                uint64_t time, uint32_t num_channels, int batch_size,
                uint32_t kernel_number, size_t input_height, size_t input_width,
                size_t kernel_height, size_t kernel_width, size_t padding_size,
                size_t stride_size) {
#ifndef PRINT_ALL
  if (mf == MeasurementFunction::CONV)
#endif
  {
    cout << setw(name_space) << impl_name << " " << setw(conv_type_space)
         << convolution_name(ct) << " :: "
#ifdef PRINT_ALL
         << setw(9) << measurement_function_name(mf) << " "
#endif
         << setw(time_space) << time << " " << setw(num_channels_space)
         << num_channels << " " << setw(batch_size_space) << batch_size << " "
         << setw(kernel_number_space) << kernel_number << " "
         << setw(input_height_space) << input_height << " "
         << setw(input_width_space) << input_width << " "
         << setw(kernel_height_space) << kernel_height << " "
         << setw(kernel_width_space) << kernel_width << " "
         << setw(padding_size_space) << padding_size << " "
         << setw(stride_size_space) << stride_size << endl;
  }
}

void bench(Registry r) {
  const int batch_size = 2;
  const float relu_alpha = 0.1;

  cout << setw(name_space) << "name"
       << " " << setw(conv_type_space) << "ct"
       << " :: "
#ifdef PRINT_ALL
       << setw(9) << "fn"
       << " "
#endif
       << setw(time_space) << "time"
       << " " << setw(num_channels_space) << "c"
       << " " << setw(batch_size_space) << "b"
       << " " << setw(kernel_number_space) << "kn"
       << " " << setw(input_height_space) << "h"
       << " " << setw(input_width_space) << "w"
       << " " << setw(kernel_height_space) << "kh"
       << " " << setw(kernel_width_space) << "kw"
       << " " << setw(padding_size_space) << "p"
       << " " << setw(stride_size_space) << "s" << endl;

  for (auto impl : r.implementations()) {
    for (auto bc : bench_cases) {
      for (auto conv_type : convolution_types) {
        auto data =
            BenchData(conv_type, batch_size, bc.num_channels, bc.kernel_number,
                      {bc.input_height, bc.input_width},
                      {bc.kernel_height, bc.kernel_width},
                      {bc.padding_size, bc.padding_size},
                      {bc.stride_size, bc.stride_size}, relu_alpha);

        auto intervals = one_run(impl, data);
        auto averages = average(intervals);
        for (auto avg : averages)
          print_line(impl.name, conv_type, avg.first, avg.second,
                     bc.num_channels, batch_size, bc.kernel_number,
                     bc.input_height, bc.input_width, bc.kernel_height,
                     bc.kernel_width, bc.padding_size, bc.stride_size);
      }
    }
  }
}
