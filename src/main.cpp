#include "bench.hpp"
#include "impl.hpp"
#include "impl/baseline_nchw/tab.hpp"
#include "impl/baseline_nhwc/tab.hpp"
#include "impl/baseline_original/tab.hpp"
#include "impl/indirect_nhwc/tab.hpp"
#include "impl/merge_im2row_ternarize/tab.hpp"
#include "impl/merge_im2row_ternarize_prelu/tab.hpp"
#include "impl/more_indirect_nhwc/tab.hpp"
#include "impl/more_indirect_prelu_nhwc/tab.hpp"
#include "impl/optmerge_im2row_ternarize/tab.hpp"
#include "verify.hpp"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

int main(int argc, char *argv[]) {
  vector<Implementation> impls = {
      {"optmerge_im2row_ternarize", DataOrder::NHWC,
       optmerge_im2row_ternarize::conv},
      {"merge_im2row_ternarize", DataOrder::NHWC, merge_im2row_ternarize::conv},
      {"merge_im2row_ternarize_prelu", DataOrder::NHWC,
       merge_im2row_ternarize_prelu::conv},
      {"indirect_nhwc", DataOrder::NHWC, indirect_nhwc::conv},
      {"more_indirect_prelu_nhwc", DataOrder::NHWC,
       more_indirect_prelu_nhwc::conv},
      {"more_indirect_nhwc", DataOrder::NHWC, more_indirect_nhwc::conv},
      {"baseline_original", DataOrder::NCHW, baseline_original::conv},
      {"indirect_nhwc", DataOrder::NHWC, indirect_nhwc::conv},
      {"baseline_nhwc", DataOrder::NHWC, baseline_nhwc::conv},
      {"baseline_nchw", DataOrder::NCHW, baseline_nchw::conv},
  };
  vector<string> filter;
  bool test = false;
  bool measure = false;

  stringstream f;
  fstream p;
  string segment;
  string bench_out = "benchmark.csv";

  InfraParameters param(0, 0, 0, 0, 0, 0, 0, 0, 0);
  string __str;
  vector<InfraParameters> params;

  int opt;
  // : means the previous option requires an argument
  while ((opt = getopt(argc, argv, "i:p:o:thb")) != -1) {
    switch (opt) {
    case 't':
      test = true;
      break;
    case 'b':
      measure = true;
      break;
    case 'h':
      cerr << "USAGE " << argv[0] << ":" << endl;
      cerr << "\t-i <impl,>\t\tRun only on the sleected implementations" << endl
           << "\t\t\t\tMultiple can be specified separated by a comma." << endl;
      cerr << "\t-p <params.csv>\t\tProvide parameters for the benchmark"
           << endl;
      cerr << "\t-o <out.csv>\t\tSpecify the output locatin for the benchark "
              "data"
           << endl;
      cerr << "\t-t\t\t\tEnable testing" << endl;
      cerr << "\t-b\t\t\tEnable benchmarking" << endl;
      exit(0);
      break;
    case 'i':
      f = stringstream(optarg);
      assert(!f.fail());

      while (std::getline(f, segment, ',')) {
        filter.push_back(segment);
      }
      break;
    case 'p':
      p = fstream(optarg);
      assert(!p.fail());

      // Ignore the first line
      if (!getline(p, __str)) {
        cerr << "Error reading csv file header" << endl;
        return 1;
      }

      while (p >> param) {
        params.push_back(param);
      };
      p.close();
      break;
    case 'o':
      bench_out = string(optarg);
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
      bench(r, params.size() == 0 ? nullptr : &params, bench_out);
  }
  return 0;
}
