#include "measure.hpp"
#include "tsc.hpp"

#include <cassert>
#include <map>

map<MeasurementEvent, string> __mes = {{MeasurementEvent::START, "START"},
                                       {MeasurementEvent::END, "END"}};

string measure_event_name(MeasurementEvent me) { return __mes[me]; }

MeasurementPoint::MeasurementPoint(string func, MeasurementEvent event,
                                   uint64_t time)
    : func(func), event(event), time(time) {}

Interval::Interval(string func, uint64_t start_time, uint64_t end_time)
    : func(func), start_time(start_time), end_time(end_time) {}

Measure *Measure::instance = nullptr;

Measure *Measure::get_instance() {
  if (instance == nullptr) {
    instance = new Measure();
  }
  return instance;
}

Measure::Measure() : measurements({}) {
  measurements.reserve(25 * measurement_event_types.size());
}

void Measure::track_memory(size_t bytes) { mem += bytes; }

size_t Measure::memory() { return mem; }

void Measure::track(const string func, const MeasurementEvent event) {
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
    while (it != ms.end()) {
      if (it->event == MeasurementEvent::START)
        break;
      it = next(it);
    }
    // this means we didn't find a start event but we still have some
    // measurements. Ultimately, this points out that some measurement calls
    // are missing
    assert(it != ms.end());
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
    auto end_point = it;

    assert(end_point != ms.end());
    assert(start_point->func == end_point->func);

    auto interval =
        Interval(start_point->func, start_point->time, end_point->time);
    intervals.push_back(interval);
    ms.erase(end_point);
    ms.erase(start_point);
  }

  return intervals;
}

void Measure::reset() {
  measurements.clear();
  mem = 0;
}
