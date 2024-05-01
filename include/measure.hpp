#pragma once

#include "common.hpp"

#include <cstdint>
#include <vector>

using namespace std;

enum class MeasurementEvent : uint8_t { START, END };
const std::array<MeasurementEvent, 2> measurement_event_types = {
    MeasurementEvent::START, MeasurementEvent::END};
string measurement_event_name(MeasurementEvent me);

enum class MeasurementFunction : uint8_t {
  TERNARIZE,
  BINARIZE,
  BTN_CNT,

  IMG2ROW,

  TNN_GEMM,
  TBN_GEMM,
  BTN_GEMM,
  BNN_GEMM,

  ALLOC,
  ALLOC2,
  FREE,

  PRELU,

  CONV,
};
const std::array<MeasurementFunction, 10> measurement_function_types = {
    MeasurementFunction::TERNARIZE, MeasurementFunction::BINARIZE,
    MeasurementFunction::BTN_CNT,   MeasurementFunction::IMG2ROW,
    MeasurementFunction::TNN_GEMM,  MeasurementFunction::TBN_GEMM,
    MeasurementFunction::BTN_GEMM,  MeasurementFunction::BNN_GEMM,
    MeasurementFunction::PRELU,     MeasurementFunction::CONV,
};
string measurement_function_name(MeasurementFunction mf);

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
  MeasurementFunction func;
  MeasurementEvent event;
  uint64_t time;

  MeasurementPoint(MeasurementFunction func, MeasurementEvent event,
                   uint64_t time);
};

class Interval {
public:
  MeasurementFunction func;
  uint64_t start_time;
  uint64_t end_time;

  Interval(MeasurementFunction func, uint64_t start_time, uint64_t end_time);
};

class Measure {
private:
  static Measure *instance;
  vector<MeasurementPoint> measurements;

public:
  Measure();
  static Measure *get_instance();

  void track(MeasurementFunction func, MeasurementEvent event);
  vector<Interval> intervals();
  void reset();
};
