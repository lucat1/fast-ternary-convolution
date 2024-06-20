"""Makes a plot from CSV."""

import matplotlib.pyplot as plt
from math import log2
from plotting.datatypes import ConvType,Function
# from plotting.opcount.util import get_work_for_function, get_data_movement_for_function
from plotting.impls.baseline import Baseline
from plotting.utils import set_plot_params,unzip_data_points,get_batch_size,get_input_size
from plotting.machine_info import get_machine_info
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

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

machine = get_machine_info()

# Specific to AMD Ryzen 7 PRO 7840U w/ Radeon 780M Graphics.
smembw = 51.5 * 10e9 # bytes/s

print(f'machine.base_frequency = {machine.base_frequency}:{type(machine.base_frequency)}')

cycles = machine.base_frequency * 10e9 # cycles/s
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

def sanity_check_df(benchmark_df: pd.DataFrame):
    df_column_set = set(benchmark_df.columns)
    correct_columns_set = set(csv_columns)
    assert correct_columns_set.issubset(df_column_set), f"The csv does not have the correct columns. Columns should be {csv_columns}"


def create_roofline(benchmark_dir: Path, output_dir: Path,verbose:bool) -> None:
    if not benchmark_dir.exists() and benchmark_dir.is_dir():
        raise ValueError(f"Benchmark directory at {benchmark_dir.absolute()} does not exist")
    output_dir.mkdir(exist_ok=True,parents=True)
    benchmark_files = [file for file in benchmark_dir.iterdir() if file.suffix == ".csv"]
    
    machine = get_machine_info()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot()

    for benchmark_file in benchmark_files:
        benchmark_df = pd.read_csv(benchmark_file)
        sanity_check_df(benchmark_df)
        experiment_names = benchmark_df["name"].unique()
        if benchmark_df.empty:
            if verbose:
                print(f"No data found for benchmark suite {benchmark_file.name}")
            continue
        conv_type_dir = PLOT_DIR / benchmark_file.name[:-4]
        conv_type_dir.mkdir(exist_ok=True)
        mask = benchmark_df['fn'].isin(list(map(lambda x: x.value, ignored_functions)))
        filtered_functions = benchmark_df[~mask]
        functions = [Function[func.upper()] for func in filtered_functions['fn'].unique()]
        for function in functions:
            df_by_func = benchmark_df[benchmark_df["fn"] == function.value]
            if df_by_func.empty:
                if verbose:
                    print(f"No data found for function {function} in benchmark suite {benchmark_file.name}")
                continue
            plt.clf()
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            ax = fig.add_subplot()
            for exp_name in experiment_names:
                df_by_func_and_exp = df_by_func[df_by_func["name"] == exp_name]
                if df_by_func_and_exp.empty:
                    if verbose:
                        print(f"No data found for experiment {exp_name}, function {function} and Benchmark suite {benchmark_file.name}")
                    continue
                if verbose:
                    print(f'Plotting data for Experiment {exp_name}, Benchmark suite {benchmark_file.name} and Function {function.value}')
                xs, ys = [], []
                for _,data_point in df_by_func_and_exp.iterrows():
                    # print(data_point)
                    cost = Baseline(data_point).cost()
                    iops, flops = cost.iops, cost.flops
                    q = cost.q
                    cycles = data_point.cycles

                    I = (iops+flops)/q
                    P = (iops+flops)/cycles
                    print(f"{function.value} I = {iops+flops}/{q} = {I}")
                    print(f"{function.value} P = {iops+flops}/{cycles} = {P}")

                    xs.append(I)
                    ys.append(P)

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

                    print(f"memory bound range for {name}: {pf/beta}")
                    memboundrange = np.arange((.05)/beta, pf/beta+0.005, step=0.0001)
                    plt.loglog(memboundrange, [beta*i for i in memboundrange], color='black', base=2, linestyle=style)

                ax.set_xscale('log', base=2)
                ax.set_yscale('log', base=2)
                assert(len(xs) == len(ys))
                xs, ys = unzip_data_points(xs,ys)
                if len(xs) < 2:
                    ax.scatter(xs,ys, label=exp_name)
                else:
                    ax.plot(xs, ys, label=exp_name)
                ax.set_xlabel("Operational Intensity [ops/byte]")
                ax.set_ylabel('Performance [ops/cycle]',
                        rotation='horizontal',
                        loc='top',
                        labelpad=-112)
            # set_plot_params(ax, machine, conv_type_dir / function.value, function.fancy())
            ax.legend()
            set_plot_params(ax, machine, conv_type_dir / function.value, function.fancy(), benchmark_file.name[:-4])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Plotter",
                                     description="Plot the benchmarks for ASL")
    parser.add_argument("-i","--input",default=str(DATA_DIR))
    parser.add_argument("-o","--output",default=str(PLOT_DIR))
    parser.add_argument('-v', '--verbose',
                    action='store_true')
    args = parser.parse_args()
    print(args)
    create_roofline(Path(args.input), Path(args.output), args.verbose)
