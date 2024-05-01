#include "measure.hpp"
#include "tsc.hpp"

#include <cassert>
#include <map>

map<MeasurementEvent, string> __mes = {{MeasurementEvent::START, "START"},
                                       {MeasurementEvent::END, "END"}};

string measure_event_name(MeasurementEvent me) { return __mes[me]; }

map<MeasurementFunction, string> __mfs = {
    {MeasurementFunction::TERNARIZE, "TERNARIZE"},
    {MeasurementFunction::BINARIZE, "BINARIZE"},
    {MeasurementFunction::BTN_CNT, "BTN_CNT"},
    {MeasurementFunction::IMG2ROW, "IMG2ROW"},
    {MeasurementFunction::TNN_GEMM, "TNN_GEMM"},
    {MeasurementFunction::TBN_GEMM, "TBN_GEMM"},
    {MeasurementFunction::BTN_GEMM, "BTN_GEMM"},
    {MeasurementFunction::BNN_GEMM, "BNN_GEMM"},
    {MeasurementFunction::PRELU, "PRELU"},
    {MeasurementFunction::CONV, "CONV"},

};

string measurement_function_name(MeasurementFunction mf) { return __mfs[mf]; }

MeasurementPoint::MeasurementPoint(MeasurementFunction func,
                                   MeasurementEvent event, uint64_t time)
    : func(func), event(event), time(time) {}

Interval::Interval(MeasurementFunction func, uint64_t start_time,
                   uint64_t end_time)
    : func(func), start_time(start_time), end_time(end_time) {}

Measure *Measure::instance = nullptr;

Measure *Measure::get_instance() {
  if (instance == nullptr) {
    instance = new Measure();
  }
  return instance;
}

Measure::Measure() : measurements({}) {
  measurements.reserve(measure_event_types.size() *
                       measurement_function_types.size());
}

void Measure::track(MeasurementFunction func, MeasurementEvent event) {
  uint64_t time;
  switch (event) {
  case MeasurementEvent::START:
    time = read_start();
    break;
  case MeasurementEvent::END:
    time = read_end();
    break;
  default:
    // Just to make clang happy. Will never occour actually
    time = 0;
    assert(0);
  };
  auto point = MeasurementPoint(func, event, time);
  // This should never result in an extra allocation
  measurements.push_back(point);
}

vector<Interval> Measure::intervals() {
  vector<Interval> intervals({});
  // This should be enough so that intervals is never re-allocated
  intervals.reserve(measurements.size() / 2);

  vector<MeasurementPoint> ms(measurements);
  while (ms.size() > 0) {
    // find the first starting point
    auto it = ms.begin();
    while (it->event != MeasurementEvent::START)
      it = next(it);
    // this means we didn't find a start event but we still have some
    // measurements. Ultimately, this points out that some measurement calls
    // are missing
    if (it == ms.end())
      assert(0);
    auto start_point = it;

    // find a mathing end point
    it = next(it);
    while (it != ms.end()) {
      if (it->func == start_point->func && it->event == MeasurementEvent::END)
        break;
      it = next(it);
    }
    // this means we didn't find a matching end measurement. Ultimately, this
    // points out that some measurement calls are missing
    if (it == ms.end())
      assert(0);
    auto end_point = it;

    assert(start_point->func == end_point->func);
    auto interval =
        Interval(start_point->func, start_point->time, end_point->time);
    intervals.push_back(interval);
    ms.erase(start_point);
    ms.erase(end_point);
  }

  return intervals;
}

void Measure::reset() { measurements.clear(); }
