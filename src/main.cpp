#include "bench.hpp"
#include "common.hpp"
#include "impl.hpp"
#include "impl/all_opts_merged/tab.hpp"
#include "impl/indirect/tab.hpp"
#include "impl/more_indirect/tab.hpp"
#include "impl/nchw/tab.hpp"
#include "impl/nchw_tmacro1/tab.hpp"
#include "impl/nchw_tmacro1_sinline/tab.hpp"
#include "impl/nchw_tmacro2/tab.hpp"
#include "impl/nchw_tmacro2_sinline/tab.hpp"
#include "impl/nhwc/tab.hpp"
#include "impl/nhwc_tmacro1/tab.hpp"
#include "impl/nhwc_tmacro1_sinline/tab.hpp"
#include "impl/nhwc_tmacro2/tab.hpp"
#include "impl/nhwc_tmacro2_sinline/tab.hpp"
#include "impl/original/tab.hpp"
#include "impl/t2r_gemmLU/tab.hpp"
#include "impl/t2r_gemmLU_autoblock/tab.hpp"
#include "impl/t2r_gemmLU_block/tab.hpp"
#include "impl/t2r_gemmLU_unroll/tab.hpp"
#include "impl/t2r_gemmLU_lord/tab.hpp"
#include "impl/tern2row/tab.hpp"
#include "impl/tern2row_cpy/tab.hpp"
#include "impl/tern2row_memcpy/tab.hpp"
#include "impl/ternary_nhwc/tab.hpp"
#include "verify.hpp"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

int main(int argc, char *argv[]) {
  cout << setw(impl_name_space) << "asserts"
       << " :: "
#ifdef NDEBUG
       << "disabled" << endl;
#else
       << "enabled" << endl;
#endif
  cout << setw(impl_name_space) << "inlines"
       << " :: "
#ifdef INLINE
       << "enabled" << endl;
#else
       << "disabled" << endl;
#endif
  cout << setw(impl_name_space) << "internal measurement"
       << " :: "
#ifdef MEASURE_INTERNAL
       << "enabled" << endl;
#else
       << "disabled" << endl;
#endif

  // TODO: Double check that we actually register different functions
  vector<Implementation> impls = {
      {"all_opts_merged", DataOrder::NHWC, all_opts_merged::conv},

            {"t2r_gemmLU_unroll", DataOrder::NHWC, t2r_gemmLU_unroll::conv},
      {"t2r_gemmLU_block", DataOrder::NHWC, t2r_gemmLU_block::conv},
      {"t2r_gemmLU_lord", DataOrder::NHWC, t2r_gemmLU_lord::conv},
      {"t2r_gemmLU", DataOrder::NHWC, t2r_gemmLU::conv},
      {"tern2row_memcpy", DataOrder::NHWC, tern2row_memcpy::conv},
      {"tern2row_cpy", DataOrder::NHWC, tern2row_cpy::conv},
      {"tern2row", DataOrder::NHWC, tern2row::conv},
      
      {"more_indirect", DataOrder::NHWC, more_indirect::conv},
      {"indirect", DataOrder::NHWC, indirect::conv},

      {"nhwc_tmacro2_sinline", DataOrder::NHWC, nhwc_tmacro2_sinline::conv},
      {"nhwc_tmacro1_sinline", DataOrder::NHWC, nhwc_tmacro1_sinline::conv},
      {"nhwc_tmacro2", DataOrder::NHWC, nhwc_tmacro2::conv},
      {"nhwc_tmacro1", DataOrder::NHWC, nhwc_tmacro1::conv},
      {"ternary_nhwc", DataOrder::NHWC, ternary_nhwc::conv},
      {"nhwc", DataOrder::NHWC, nhwc::conv},
      
      {"nchw_tmacro2_sinline", DataOrder::NCHW, nchw_tmacro2_sinline::conv},
      {"nchw_tmacro1_sinline", DataOrder::NCHW, nchw_tmacro1_sinline::conv},
      {"nchw_tmacro2", DataOrder::NCHW, nchw_tmacro2::conv},
      {"nchw_tmacro1", DataOrder::NCHW, nchw_tmacro1::conv},
      {"nchw", DataOrder::NCHW, nchw::conv},
      
      {"original", DataOrder::NCHW, original::conv},
  };
  vector<string> filter;
  bool test = false;
  bool measure = false;
  bool convonly = false;

  stringstream f;
  fstream p;
  string segment;
  string bench_out = "benchmark.csv";

  InfraParameters param(0, 0, 0, 0, 0, 0, 0, 0, 0);
  string __str;
  vector<InfraParameters> params;

  int opt;
  // : means the previous option requires an argument
  while ((opt = getopt(argc, argv, "i:p:o:thbc")) != -1) {
    switch (opt) {
    case 't':
      test = true;
      break;
    case 'b':
      measure = true;
      break;
    case 'c':
      convonly = true;
      break;
    case 'h':
      cerr << "USAGE " << argv[0] << ":" << endl;
      cerr << "\t-i <impl,>\t\tRun only on the selected implementations" << endl
           << "\t\t\t\tMultiple can be specified separated by a comma." << endl;
      cerr << "\t-p <params.csv>\t\tProvide parameters for the benchmark"
           << endl;
      cerr << "\t-o <out.csv>\t\tSpecify the output locatin for the benchark "
              "data"
           << endl;
      cerr << "\t-t\t\t\tEnable testing" << endl;
      cerr << "\t-b\t\t\tEnable benchmarking" << endl;
      cerr << "\t-c\t\t\tOnly print benchmark data for the whole convolution"
           << endl;
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
      bench(r, params.size() == 0 ? nullptr : &params, bench_out, convonly);
  }
  return 0;
}
