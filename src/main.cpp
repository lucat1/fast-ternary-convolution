#include "bench.hpp"
#include "impl.hpp"
#include "verify.hpp"

#include <algorithm>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <vector>

#include "impl/baseline/tab.hpp"

using namespace std;

int main(int argc, char *argv[]) {
  vector<Implementation> impls = {{"baseline", baseline::conv}};
  vector<string> filter;
  bool test = false;
  bool measure = false;

  stringstream f;
  string segment;

  int opt;
  // : means the previous option requires an argument
  while ((opt = getopt(argc, argv, "i:thb")) != -1) {
    switch (opt) {
    case 't':
      test = true;
      break;
    case 'b':
      measure = true;
      break;
    case 'h':
      cout << "USAGE " << argv[0] << ":" << endl;
      cout << "\t-i <impl,>\t\tRun only on the sleected implementations" << endl
           << "\t\t\t\tMultiple can be specified separated by a comma." << endl;
      cout << "\t-t\t\t\tEnable testing" << endl;
      cout << "\t-b\t\t\tEnable benchmarking" << endl;
      exit(0);
      break;
    case 'i':
      f = stringstream(optarg);
      while (std::getline(f, segment, ',')) {
        filter.push_back(segment);
      }
      break;
    default:
      break;
    }
  }

  Registry r;
  for (auto impl : impls)
    if (filter.size() == 0 ||
        std::find(filter.begin(), filter.end(), impl.name) != filter.end())
      r.add(impl);

  if (r.implementations().size() > 0) {
    measure_overhead();
    if (test)
      verify(r);
    if (measure)
      bench(r);
  }
  return 0;
}
