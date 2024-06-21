#include "bench.hpp"
#include "common.hpp"
#include "impl.hpp"
#include "main_impls/best_impl_avx2/tab.hpp"
#include "main_impls/best_impl_avx512/tab.hpp"

#include "minor_impls/avx2/tab.hpp"
#include "minor_impls/avx2_lessunpack/tab.hpp"
#include "minor_impls/avx2_lessunpack_popout/tab.hpp"
#include "minor_impls/avx2_popout/tab.hpp"

#include "main_impls/data_order_nhwc/tab.hpp"
#include "main_impls/data_order_nhwc_tensor_macro1/tab.hpp"
#include "main_impls/original/tab.hpp"
#include "minor_impls/indirect/tab.hpp"
#include "minor_impls/more_indirect/tab.hpp"
#include "minor_impls/nchw/tab.hpp"
#include "minor_impls/nchw_tmacro1/tab.hpp"
#include "minor_impls/nchw_tmacro1_sinline/tab.hpp"
#include "minor_impls/nchw_tmacro2/tab.hpp"
#include "minor_impls/nchw_tmacro2_sinline/tab.hpp"
#include "minor_impls/nhwc_tmacro1_sinline/tab.hpp"
#include "minor_impls/nhwc_tmacro2/tab.hpp"
#include "minor_impls/nhwc_tmacro2_sinline/tab.hpp"

#include "minor_impls/t2r_avx2u_gemmLU_block/tab.hpp"
#include "minor_impls/t2r_avx2u_permute_gemmLU_block/tab.hpp"
#include "minor_impls/t2r_avx2u_permute_ur_gemmLU_block/tab.hpp"
#include "minor_impls/t2r_avx2u_ur_gemmLU_block/tab.hpp"
#include "minor_impls/t2r_avx512u_gemmLU_block/tab.hpp"
#include "minor_impls/t2r_avx512u_ur_gemmLU_block/tab.hpp"

#include "main_impls/t2r_gemmLU/tab.hpp"
#include "minor_impls/t2r_gemmLU_autoblock/tab.hpp"
#include "minor_impls/t2r_gemmLU_block/tab.hpp"
#include "minor_impls/t2r_gemmLU_block_avx2/tab.hpp"
#include "minor_impls/t2r_gemmLU_block_avx512/tab.hpp"
#include "minor_impls/t2r_gemmLU_lord/tab.hpp"
#include "minor_impls/t2r_gemmLU_unroll/tab.hpp"
#include "minor_impls/t2r_ur_gemmLU_block/tab.hpp"
#include "minor_impls/tern2row/tab.hpp"
#include "minor_impls/tern2row_cpy/tab.hpp"
#include "minor_impls/tern2row_memcpy/tab.hpp"
#include "minor_impls/ternary_nhwc/tab.hpp"
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

  vector<Implementation> impls = {
      {"t2r_avx2u_gemmLU_block", DataOrder::NHWC, t2r_avx2u_gemmLU_block::conv},
      {"t2r_avx2u_permute_gemmLU_block", DataOrder::NHWC,
       t2r_avx2u_permute_gemmLU_block::conv},
      {"t2r_avx2u_permute_ur_gemmLU_block", DataOrder::NHWC,
       t2r_avx2u_permute_ur_gemmLU_block::conv},
      {"t2r_avx2u_ur_gemmLU_block", DataOrder::NHWC,
       t2r_avx2u_ur_gemmLU_block::conv},
      {"t2r_avx512u_gemmLU_block", DataOrder::NHWC,
       t2r_avx512u_gemmLU_block::conv},
      {"t2r_avx512u_ur_gemmLU_block", DataOrder::NHWC,
       t2r_avx512u_ur_gemmLU_block::conv},

      {"best_impl_avx512", DataOrder::NHWC, best_impl_avx512::conv},
      {"best_impl_avx2", DataOrder::NHWC, best_impl_avx2::conv},

      {"avx2_lessunpack_popout", DataOrder::NHWC, avx2_lessunpack_popout::conv},
      {"avx2_popout", DataOrder::NHWC, avx2_popout::conv},
      {"avx2_lessunpack", DataOrder::NHWC, avx2_lessunpack::conv},
      {"avx2", DataOrder::NHWC, avx2::conv},

      {"t2r_gemmLU_unroll", DataOrder::NHWC, t2r_gemmLU_unroll::conv},
      {"t2r_gemmLU", DataOrder::NHWC, t2r_gemmLU::conv},
      {"t2r_gemmLU_block", DataOrder::NHWC, t2r_gemmLU_block::conv},
      {"t2r_gemmLU_autoblock", DataOrder::NHWC, t2r_gemmLU_autoblock::conv},
      {"t2r_ur_gemmLU_block", DataOrder::NHWC, t2r_ur_gemmLU_block::conv},
      {"t2r_gemmLU_block_avx2", DataOrder::NHWC, t2r_gemmLU_block_avx2::conv},
      {"t2r_gemmLU_block_avx512", DataOrder::NHWC,
       t2r_gemmLU_block_avx512::conv},
      {"t2r_gemmLU_lord", DataOrder::NHWC, t2r_gemmLU_lord::conv},
      {"tern2row_memcpy", DataOrder::NHWC, tern2row_memcpy::conv},
      {"tern2row_cpy", DataOrder::NHWC, tern2row_cpy::conv},
      {"tern2row", DataOrder::NHWC, tern2row::conv},

      {"more_indirect", DataOrder::NHWC, more_indirect::conv},
      {"indirect", DataOrder::NHWC, indirect::conv},

      {"nhwc_tmacro2_sinline", DataOrder::NHWC, nhwc_tmacro2_sinline::conv},
      {"nhwc_tmacro1_sinline", DataOrder::NHWC, nhwc_tmacro1_sinline::conv},
      {"nhwc_tmacro2", DataOrder::NHWC, nhwc_tmacro2::conv},
      {"data_order_nhwc_tensor_macro1", DataOrder::NHWC,
       data_order_nhwc_tensor_macro1::conv},
      {"ternary_nhwc", DataOrder::NHWC, ternary_nhwc::conv},
      {"data_order_nhwc", DataOrder::NHWC, data_order_nhwc::conv},

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
  bool hot_cache = true;

  stringstream f;
  fstream p;
  string segment;
  string bench_out = "benchmark.csv";

  InfraParameters param(0, 0, 0, 0, 0, 0, 0, 0, 0);
  string __str;
  vector<InfraParameters> params;

  int opt;
  // : means the previous option requires an argument
  while ((opt = getopt(argc, argv, "i:p:o:thbcl")) != -1) {
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
      cerr << "\t-l\t\t\tUse data that is cold in cache" << endl;
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
    case 'l':
      hot_cache = false;
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
      bench(r, params.size() == 0 ? nullptr : &params, bench_out, convonly,
            hot_cache);
  }
  return 0;
}
