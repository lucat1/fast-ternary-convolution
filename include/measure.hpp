#pragma once

#include "common.hpp"

#include <cstdint>
#include <vector>

using namespace std;

namespace measurement_point {

static const string conv = "conv";
static const string ternarize = "ternarize";
static const string im2row = "im2row";
static const string ternarize_im2row = "terna2row";
static const string gemm = "gemm";
static const string im2rowgemm = "im2rowgemm";
static const string prelu = "prelu";
static const string gemmprelu = "gemmprelu";

static const string alloc = "alloc";

} // namespace measurement_point

enum class MeasurementEvent : uint8_t { START, END };
const std::array<MeasurementEvent, 2> measurement_event_types = {
    MeasurementEvent::START, MeasurementEvent::END};
string measurement_event_name(MeasurementEvent me);

#ifdef MEASURE_INTERNAL
#define measure_point(measurement_func, measurement_type)                      \
  Measure::get_instance()->track(measurement_func, measurement_type)
#else
// This is equivalent to a NOOP
#define measure_point(measurement_func, measurement_type)                      \
  do {                                                                         \
  } while (0);
#endif

class MeasurementPoint {
public:
  string func;
  MeasurementEvent event;
  uint64_t time;

  MeasurementPoint(string func, MeasurementEvent event, uint64_t time);
};

class Interval {
public:
  string func;
  uint64_t start_time;
  uint64_t end_time;

  Interval(string func, uint64_t start_time, uint64_t end_time);
};

class Measure {
private:
  static Measure *instance;
  vector<MeasurementPoint> measurements;
  size_t mem;

public:
  Measure();
  static Measure *get_instance();

  void track(const string func, const MeasurementEvent event);
  void track_memory(size_t bytes);
  vector<Interval> intervals();
  size_t memory();
  void reset();
};
