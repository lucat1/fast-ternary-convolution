#include "measure.hpp"
#include "tsc.hpp"

#include <cassert>

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
  measurements.reserve(static_cast<size_t>(MeasurementEvent::__count) *
                       static_cast<size_t>(MeasurementFunction::__count));
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
  case MeasurementEvent::__count:
    assert(0);
    break;
  default:
    // make pendantic mode happy
    time = 0;
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
    while (it->event != MeasurementEvent::END && it->func == start_point->func)
      it = next(it);
    // this means we didn't find a matching end measurement. Ultimately, this
    // points out that some measurement calls are missing
    if (it == ms.end())
      assert(0);
    auto end_point = it;

    auto interval =
        Interval(start_point->func, start_point->time, end_point->time);
    intervals.push_back(interval);
    ms.erase(start_point);
    ms.erase(end_point);
  }

  return intervals;
}

void Measure::reset() { measurements.clear(); }
