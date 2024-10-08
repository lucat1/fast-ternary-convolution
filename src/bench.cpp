#include "bench.hpp"
#include "common.hpp"
#include "measure.hpp"
#include "problem_data.hpp"
#include "tsc.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>

// uncomment this for a detailed (visual) performance measurement
#define PRINT_ALL

using namespace std;

static size_t overhead = 0;

std::vector<InfraParameters> bench_cases = {
    {64, 1, 56, 56, 64, 3, 3, 1, 1},    {64, 1, 56, 56, 128, 3, 3, 1, 1},
    {128, 1, 28, 28, 128, 3, 3, 1, 1},  {128, 1, 28, 28, 256, 3, 3, 1, 1},
    {256, 1, 14, 14, 256, 3, 3, 1, 1},  {256, 1, 14, 14, 512, 3, 3, 1, 1},
    {80, 1, 224, 224, 80, 3, 3, 1, 1},  {80, 1, 224, 224, 80, 3, 3, 1, 2},
    {80, 1, 224, 224, 80, 3, 3, 1, 3},  {80, 1, 224, 224, 80, 3, 3, 1, 4},
    {512, 1, 56, 56, 256, 1, 1, 0, 1},  {512, 1, 56, 56, 256, 3, 3, 1, 1},
    {512, 1, 56, 56, 256, 5, 5, 2, 1},  {512, 1, 56, 56, 256, 7, 7, 3, 1},
    {512, 1, 56, 56, 256, 9, 9, 3, 1},  {512, 1, 56, 56, 256, 11, 11, 3, 1},
    {2000, 1, 1, 1, 4000, 1, 1, 0, 1},  {4000, 1, 1, 1, 8000, 1, 1, 0, 1},
    {8000, 1, 1, 1, 16000, 1, 1, 0, 1}, {16000, 1, 1, 1, 32000, 1, 1, 0, 1},
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
  BenchData(ConvolutionType conv_type, DataOrder data_order, InfraParameters p,
            float relu_alpha)
      : Data(conv_type, data_order, p, relu_alpha) {
    distribution = std::uniform_int_distribution<int>(-1, 1);

    randomize(input.data, input.dim1 * input.dim2 * input.dim3 * input.dim4,
              has_ternary_input(conv_type));

    for (size_t i = 0; i < threshold.size; ++i)
      threshold.data[i] = 0.5;
  }
};

vector<pair<vector<Interval>, size_t>>
one_run(Implementation impl, Data &data, bool hot_cache,
        ConvolutionType conv_type, DataOrder data_order, InfraParameters p,
        float relu_alpha) {
  size_t num_runs = 20;
  vector<pair<vector<Interval>, size_t>> measurement_intervals;
  auto m = Measure::get_instance();

  if (hot_cache) {
    for (size_t i = 0; i < num_runs; ++i) {
      m->reset();

      m->track(measurement_point::conv, MeasurementEvent::START);
      impl.fn(data.input, data.threshold, data.padding_h, data.padding_w,
              data.kernel, data.stride_h, data.stride_w, data.relu_alpha);
      m->track(measurement_point::conv, MeasurementEvent::END);

      measurement_intervals.push_back({m->intervals(), m->memory()});
      m->reset();
    }
  } else {
    for (size_t i = 0; i < num_runs; ++i) {
      auto d = BenchData(conv_type, data_order, p, relu_alpha);
      memcpy(d.input.data, data.input.data,
             sizeof(float) * data.batch_size *
                 nchw_or_nhwc(data.channels, data.input_h) *
                 nchw_or_nhwc(data.input_h, data.input_w) *
                 nchw_or_nhwc(data.input_w, data.channels));
      memcpy(d.kernel.data, data.kernel.data,
             sizeof(int64_t) * data.kernel_n * data.kernel_h * data.kernel_w *
                 int64s_for_bits(data.channels) * 2);
      m->reset();

      m->track(measurement_point::conv, MeasurementEvent::START);
      impl.fn(d.input, d.threshold, d.padding_h, d.padding_w, d.kernel,
              d.stride_h, d.stride_w, d.relu_alpha);
      m->track(measurement_point::conv, MeasurementEvent::END);

      measurement_intervals.push_back({m->intervals(), m->memory()});
      m->reset();
    }
  }

  return measurement_intervals;
}

pair<map<const string, uint64_t>, size_t>
average(vector<pair<vector<Interval>, size_t>> raw) {
  map<const string, vector<Interval>> by_func;
  size_t mem = 0;

  for (auto one_run : raw) {
    for (auto interval : one_run.first) {
      auto fn = interval.func;
      if (!by_func.contains(fn))
        by_func.insert({fn, {}});
      by_func[fn].push_back(interval);
    }
    mem += one_run.second;
  }

  map<const string, uint64_t> avgs;
  for (auto event : by_func) {
    auto func = event.first;
    uint64_t tot = 0;

    for (auto interval : event.second)
      tot += interval.end_time - interval.start_time;

    // remove the measurement overhead
    tot -= overhead * event.second.size();

    uint64_t avg = tot / event.second.size();
    avgs.insert({func, avg});
  }

  return {avgs, mem / raw.size()};
}

const constexpr size_t conv_type_space = 3;
const constexpr size_t cycles_space = 12;
const constexpr size_t channels_space = 5;
const constexpr size_t batch_size_space = 2;
const constexpr size_t kernel_number_space = 4;
const constexpr size_t input_height_space = 3;
const constexpr size_t input_width_space = 3;
const constexpr size_t kernel_height_space = 2;
const constexpr size_t kernel_width_space = 2;
const constexpr size_t padding_size_space = 1;
const constexpr size_t stride_size_space = 1;

void print_line(ofstream &csv, bool convonly, size_t bytes, string impl_name,
                ConvolutionType ct, const string mf, uint64_t cycles,
                uint32_t channels, int batch_size, uint32_t kernel_number,
                size_t input_height, size_t input_width, size_t kernel_height,
                size_t kernel_width, size_t padding_size, size_t stride_size) {
  // convonly ==> mf == measurement_point::conv
  if (!convonly || mf == measurement_point::conv) {
    cout << setw(impl_name_space) << impl_name << " " << setw(conv_type_space)
         << convolution_name(ct) << " :: ";
    if (!convonly)
      cout << setw(9) << mf << " ";

    cout << setw(cycles_space) << cycles << " " << setw(channels_space)
         << channels << " " << setw(batch_size_space) << batch_size << " "
         << setw(kernel_number_space) << kernel_number << " "
         << setw(input_height_space) << input_height << " "
         << setw(input_width_space) << input_width << " "
         << setw(kernel_height_space) << kernel_height << " "
         << setw(kernel_width_space) << kernel_width << " "
         << setw(padding_size_space) << padding_size << " "
         << setw(stride_size_space) << stride_size << endl;
  }

  csv << impl_name << "," << convolution_name(ct) << "," << mf << "," << cycles
      << "," << channels << "," << batch_size << "," << kernel_number << ","
      << input_height << "," << input_width << "," << kernel_height << ","
      << kernel_width << "," << padding_size << "," << stride_size << ","
      << bytes << endl;
}

void bench(Registry r, vector<InfraParameters> *params, string output,
           bool convonly, bool hot_cache) {
  const float relu_alpha = 0.1;
  auto csv = ofstream(output);
  assert(!csv.fail());

  if (params == nullptr)
    params = &bench_cases;

  cout << setw(impl_name_space) << "name"
       << " " << setw(conv_type_space) << "ct"
       << " :: "
#ifdef PRINT_ALL
       << setw(9) << "fn"
       << " "
#endif
       << setw(cycles_space) << "cycles"
       << " " << setw(channels_space) << "c"
       << " " << setw(batch_size_space) << "b"
       << " " << setw(kernel_number_space) << "kn"
       << " " << setw(input_height_space) << "h"
       << " " << setw(input_width_space) << "w"
       << " " << setw(kernel_height_space) << "kh"
       << " " << setw(kernel_width_space) << "kw"
       << " " << setw(padding_size_space) << "p"
       << " " << setw(stride_size_space) << "s" << endl;

  csv << "name,ct,fn,cycles,channels,batch_size,kernel_number,input_"
         "height,input_width,kernel_height,kernel_width,padding_size,stride_"
         "size,bytes"
      << endl;

  for (auto bc : *params) {
    for (auto impl : r.implementations()) {
      // for (auto conv_type : convolution_types) {
      ConvolutionType conv_type = ConvolutionType::TNN;
      auto data = BenchData(conv_type, impl.data_order, bc, relu_alpha);

      auto intervals = one_run(impl, data, hot_cache, conv_type,
                               impl.data_order, bc, relu_alpha);
      auto averages = average(intervals);
      for (auto avg : averages.first)
        print_line(csv, convonly, averages.second, impl.name, conv_type,
                   avg.first, avg.second, bc.channels, bc.batch_size,
                   bc.kernel_number, bc.input_height, bc.input_width,
                   bc.kernel_height, bc.kernel_width, bc.padding_size,
                   bc.stride_size);
      // }
    }
    cout << std::string(impl_name_space + conv_type_space + cycles_space +
                            channels_space + batch_size_space +
                            kernel_number_space + input_height_space +
                            input_width_space + kernel_height_space +
                            kernel_width_space + padding_size_space +
                            stride_size_space + 2 * 12,
                        '-')
         << endl;
  }
  csv.close();
}

void measure_overhead() {
  auto m = Measure::get_instance();
  size_t runs = 1000000;
  size_t measurement_size = 25 * measurement_event_types.size();
  size_t cycles = 0;
  for (size_t j = 0; j < runs; ++j) {
    uint64_t start = read_start();
    for (size_t i = 0; i < measurement_size; ++i)
      measure_point(measurement_point::conv, MeasurementEvent::START);
    uint64_t end = read_end();
    cycles += end - start;
    m->reset();
  }

  cycles = overhead = (cycles) / (runs * measurement_size);
  cout << setw(impl_name_space) << "measurement overhead"
       << " :: " << cycles << " cycles/call" << endl;
}
