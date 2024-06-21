"""Makes a plot from CSV."""

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter
from plotting.datatypes import ConvType,Function
from plotting.impls.baseline import Baseline
from plotting.utils import set_plot_params,unzip_data_points,get_input_size,frequency_to_number,get_experiment_names
from plotting.machine_info import get_machine_info
from pathlib import Path
import pandas as pd
import argparse

import matplotlib
matplotlib.use('pdf')

REPO_DIR = Path(__file__).parent.parent
DATA_DIR = REPO_DIR / "benchmarks"
PLOT_DIR = REPO_DIR / "plots"

plt.rcParams['font.family'] = 'Computer Modern Roman'
plt.rcParams['text.usetex'] = True

plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
STYLES = [
    # color, marker, offst, name
    ('dodgerblue', 'D', (-50, 0), "AVX2"),
    ('deeppink', 'o', (-50, -20), "AVX512"),
    ('goldenrod', 'h', (-60, -20), "NHWC"),
    ('brown', '^', (-185, -90), "Tensor Macro"),
    ('darkcyan', 'H', (-65, -15), "Merged"),
    ('darkolivegreen', 's', (-90, -100), "Merged+Blocked"),
]
title = "Increasing Channels - Fully Connected Layers"

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

'''
Merged functions are ternarize and im2row, im2row and gemm, gemm and prelu, im2row+gemm+prelu
One line to the csc will be added
'''

def sanity_check_df(benchmark_df: pd.DataFrame):
    df_column_set = set(benchmark_df.columns)
    correct_columns_set = set(csv_columns)
    print(f'correct_columns_set = {correct_columns_set}')
    assert correct_columns_set.issubset(df_column_set), f"The csv does not have the correct columns. Columns should be {csv_columns}"

def merge_unmerged(df: pd.DataFrame) -> pd.DataFrame:
    no_group = ["fn", "cycles", "bytes"]

    def merge_fns(group):
        srcs = [s.value for _, srcs in merged_funcs.items() for s in srcs]
        df = group[~group['fn'].isin(srcs)].to_dict(orient='records')
        for dst, srcs in merged_funcs.items():
            srcs = list(map(lambda x: x.value, srcs))
            mask = group['fn'].isin(srcs)
            rows = group[mask]
            if len(rows) <= 0:
                continue
            elif len(rows) == 1:
                print("Whole group:\n", group)
                print("Filtering by: ", srcs)
                print("Filtered rows:\n", rows)
                raise Exception("Expected grouping but only one matching line found")

            cycles = rows['cycles'].sum()
            row = rows.reset_index().iloc[0].to_dict()
            row['cycles'] = cycles
            row['fn'] = dst.value

            df.append(row)

        return pd.DataFrame(df)

    cols = [col for col in csv_columns if col not in no_group]
    return df.groupby(cols).apply(merge_fns).reset_index(drop=True)


def create_plots(ax: Axes, benchmark_file: Path, main: bool) -> None:
    machine_frequency = 3.3 * 10**9
    print(f"Reading {benchmark_file}")
    benchmark_df = pd.read_csv(benchmark_file)
    sanity_check_df(benchmark_df)
    benchmark_df = merge_unmerged(benchmark_df)
    impls = get_experiment_names(benchmark_df["name"].unique())
    if benchmark_df.empty:
        print(f"No data found for benchmark suite {benchmark_file.name}")
        exit(1)
    mask = benchmark_df['fn'].isin(list(map(lambda x: x.value, ignored_functions)))
    filtered_functions = benchmark_df[~mask]
    functions = [Function[func.upper()] for func in filtered_functions['fn'].unique()]
    df_by_func = benchmark_df[benchmark_df["fn"] == "conv"]
    if df_by_func.empty:
        print(f"No data found for function {function} in benchmark suite {benchmark_file.name}")
    x_data = []
    performance_values = []
    runtime_values = []
    for impl in impls:
        experiment_data = df_by_func[df_by_func["name"] == impl]
        xs, ys_runtime, ys_performance = [], [], []
        for _, data_point in experiment_data.iterrows():
            if main:
                xs.append(get_input_size(data_point) / 2**20) # turn input size into MiB
            else:
                xs.append(get_input_size(data_point) / 2**10) # turn input size into KiB
            cost = Baseline(data_point).cost()
            ys_runtime.append(data_point.cycles / machine_frequency)
            ys_performance.append((cost.iops + cost.flops) / data_point.cycles)
        if len(xs) == 0:
            continue
        xs, ys_runtime = unzip_data_points(xs,ys_runtime)

        x_data.append(xs)
        performance_values.append(ys_performance)
        runtime_values.append(ys_runtime)
    for i, (impl, x, y) in enumerate(zip(impls,x_data,runtime_values)):
        if impl == 'data_order_nhwc' or impl == 'best_impl_avx512':
            print(impl, y)
        color, marker, offst, name = STYLES[i]
        print(f"-- Adding line for {impls[impl]} ({name})")
        ax.plot(x, y, label=impls[impl], marker=marker, color=color)
        if main:
            ax.annotate(name or impls[impl], (x[-1], y[-1]), color=color, xytext=offst, textcoords='offset points', fontsize='x-large')
    if main:
        ax.set_xlabel("Input size [MiB]")
    else:
        ax.set_xlabel("Input size [KiB]", fontsize=18)
    if main:
        ax.set_ylabel('Time [s]',
                rotation='horizontal',
                loc='top',
                labelpad=-50)

    ax.tick_params(axis='both', direction='in', which='major', pad=5)
    ax.grid(which='major', axis='y', linewidth=.5, dashes=(3,3))
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    if not main:
        ax.yaxis.tick_right()
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())

    if main:
        ax.set_title(title, fontsize=15)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Plotter",
                                     description="Plot the benchmarks for ASL")
    parser.add_argument("-i","--input",default=str(DATA_DIR))
    parser.add_argument("-o","--output",default=str(PLOT_DIR))
    args = parser.parse_args()

    output_file = Path(args.output) / "1"
    Path(args.output).mkdir(exist_ok=True,parents=True)

    benchmark_file = Path(args.input) / "incr_c.csv"
    benchmark_file2 = Path(args.input) / "fc.csv"

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()
    sub_ax = inset_axes(
        parent_axes=ax,
        loc='upper left',
        width="35%",
        height="50%",
        borderpad=1
    )
    sub_ax.set_facecolor('whitesmoke')

    create_plots(ax, benchmark_file, True)
    create_plots(sub_ax, benchmark_file2, False)

    fig.savefig(output_file, bbox_inches='tight')
