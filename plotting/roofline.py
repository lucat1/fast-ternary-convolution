"""Makes a plot from CSV."""

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from math import log2
from plotting.datatypes import ConvType,Function
from plotting.impls.baseline import Baseline
from plotting.impls.best_impl_avx2 import BestImplAVX2
from plotting.impls.best_impl_avx512 import BestImplAVX512
from plotting.impls.t2r_gemmLU import T2RGemmLU
from plotting.utils import set_plot_params,unzip_data_points,get_batch_size,get_input_size
from plotting.impl import Cost
from plotting.machine_info import get_machine_info
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

import matplotlib
matplotlib.use('pdf')

REPO_DIR = Path(__file__).parent.parent
DATA_DIR = REPO_DIR / "benchmarks"
PLOT_DIR = REPO_DIR / "plots"

plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
#plt.rcParams['font.family'] = 'New Computer Modern'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
plt.rcParams['svg.fonttype'] = 'none'

title = "Whatever"

machine = get_machine_info()

# Specific to AMD Ryzen 7 PRO 7840U w/ Radeon 780M Graphics.
smembw = 51.5 * 10e9 # bytes/s

cycles = 3.3 * 10e9 # cycles/s
int_size = 32
simd_size = 256
simd_size512 = 512

ipi = 8 # ops/cycle
fpi = 6 # ops/cycle
ipi_simd = ipi*(simd_size/int_size) # ops/cycle
fpi_simd = fpi*(simd_size/int_size) # ops/cycle
ipi_simd512 = ipi*(simd_size512/int_size) # ops/cycle
fpi_simd512 = fpi*(simd_size512/int_size) # ops/cycle
beta = smembw / cycles

conv_types = [conv_type for conv_type in ConvType]
functions = [function_type for function_type in Function]
csv_columns = ["name","ct","fn","cycles","channels",
               "batch_size","kernel_number","input_height","input_width",
               "kernel_height","kernel_width","padding_size","stride_size", "bytes"]

merged_funcs = {
    Function.TERNA2ROW: [Function.TERNARIZE, Function.IM2ROW],
    Function.GEMMPRELU: [Function.GEMM, Function.PRELU],
}
ignored_functions = [
    Function.ALLOC, Function.TERNARIZE, Function.IM2ROW
]


def cost_of_data_point(parameters: pd.Series) -> Cost:
    """
    Compute the cost of a given data point.

    This depends upon the implementation.
    """
    match parameters['name']:
        case 'best_impl_avx512':
            return BestImplAVX512(parameters).cost()
        case 'best_impl_avx2':
            return BestImplAVX2(parameters).cost()
        case 't2r_gemmLU':
            return T2RGemmLU(parameters).cost()
        case 't2r_gemmLU_block':
            return T2RGemmLU(parameters).cost()
        case _:
            return Baseline(parameters).cost()

def sanity_check_df(benchmark_df: pd.DataFrame):
    df_column_set = set(benchmark_df.columns)
    correct_columns_set = set(csv_columns)
    assert correct_columns_set.issubset(df_column_set), f"The csv does not have the correct columns. Columns should be {csv_columns}"


def create_roofline(ax: Axes, benchmark_file: Path) -> None:
    benchmark_df = pd.read_csv(benchmark_file)
    sanity_check_df(benchmark_df)
    impls = benchmark_df["name"].unique()
    if benchmark_df.empty:
        print(f"No data found for benchmark suite {benchmark_file.name}")
        exit(1)
    df_by_func = benchmark_df[benchmark_df["fn"] == "conv"]
    if df_by_func.empty:
        print(f"No data found for function conv in benchmark suite {benchmark_file.name}")
        exit(1)

    # draw the rooflines
    xmax = 5
    rng = range(0, xmax)
    for (name, pf, style) in [("\\pi_{is}", ipi, '-'),
                              ("\\pi_{fs}", fpi, "-"),
                              ("\\pi_{iv}", ipi_simd, "--"),
                              ("\\pi_{fv}", fpi_simd, "--"),
                              ("\\pi_{iv512}", ipi_simd512, "--"),
                              ("\\pi_{fv512}", fpi_simd512, "--")]:
        plt.hlines(y=pf, color='black', linestyle=style, xmin=pf/beta, xmax=xmax)
        up = 6
        plt.text(rng[-1], pf+log2(up), f"$P(n) \\leq {name}$", verticalalignment='bottom', horizontalalignment='right')

        # print(f"memory bound range for {name}: {pf/beta}")
        memboundrange = np.arange((1)/beta, pf/beta+0.005, step=0.0001)
        plt.loglog(memboundrange, [beta*i for i in memboundrange], color='black', base=2, linestyle=style)

    for impl in impls:
        df_by_func_and_exp = df_by_func[df_by_func["name"] == impl]
        if df_by_func_and_exp.empty:
            print(f"No data found for experiment {impl}, function {function} and Benchmark suite {benchmark_file.name}")
            continue
        print(f'Plotting data for Experiment {impl}, Benchmark suite {benchmark_file.name}')
        xs, ys = [], []
        for _,data_point in df_by_func_and_exp.iterrows():
            cost = cost_of_data_point(data_point)
            iops, flops, q = cost.iops, cost.flops, cost.q
            cycles = data_point.cycles

            I = (iops+flops)/q
            P = (iops+flops)/cycles
            print(f"{impl} I = {iops+flops}/{q} = {I}")
            print(f"{impl} P = {iops+flops}/{cycles} = {P}")

            xs.append(I)
            ys.append(P)

        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        assert(len(xs) == len(ys))
        xs, ys = unzip_data_points(xs,ys)
        ax.plot(xs, ys, label=impl)
        ax.set_xlabel("Operational Intensity [ops/byte]")
        ax.set_ylabel('Performance [ops/cycle]',
                rotation='horizontal',
                loc='top',
                labelpad=-112)
        ax.legend()
        ax.tick_params(axis='both', direction='in', which='major', pad=5)
        ax.grid(which='major', axis='y', linewidth=.5, dashes=(3,3))
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.set_xscale("log", base=2)

        ax.set_title(title, fontsize=15)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Plotter",
                                     description="Plot the benchmarks for ASL")
    parser.add_argument("-i","--input",default=str(DATA_DIR))
    parser.add_argument("-o","--output",default=str(PLOT_DIR))
    parser.add_argument('-v', '--verbose',
                    action='store_true')
    args = parser.parse_args()

    output_file = Path(args.output) / "roof"
    Path(args.output).mkdir(exist_ok=True,parents=True)

    benchmark_file = Path(args.input) / "incr_c.csv"

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot()
    create_roofline(ax, benchmark_file)
    print(output_file)
    plt.savefig(output_file, bbox_inches='tight')
