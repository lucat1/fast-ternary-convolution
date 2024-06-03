"""Makes a plot from CSV."""

import matplotlib.pyplot as plt
from plot.datatypes import ConvType,Function
from plot.impls.baseline import Baseline
from plot.opcount.util import get_work_for_function, get_data_movement_for_function
from plot.utils import set_plot_params,unzip_data_points,get_input_size,frequency_to_number,get_experiment_names
from plot.machine_info import get_machine_info
from pathlib import Path
import pandas as pd
import argparse

REPO_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = REPO_DIR / "benchmarks"
PLOT_DIR = REPO_DIR / "plots"

plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 8
#plt.rcParams['font.family'] = 'New Computer Modern'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
plt.rcParams['svg.fonttype'] = 'none'


conv_types = [conv_type for conv_type in ConvType]
functions = [function_type for function_type in Function]
csv_columns = ["name","ct","fn","cycles","channels",
               "batch_size","kernel_number","input_height","input_width",
               "kernel_height","kernel_width","padding_size","stride_size", "bytes"]

merged_funcs = {
    Function.TERNA2ROW: [Function.TERNARIZE, Function.IM2ROW],
#    Function.GEMMPRELU: [Function.GEMM, Function.PRELU],
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

def create_plots(benchmark_dir: Path, output_dir: Path,verbose:bool) -> None:
    if not benchmark_dir.exists() and benchmark_dir.is_dir():
        raise ValueError(f"Benchmark directory at {benchmark_dir.absolute()} does not exist")
    output_dir.mkdir(exist_ok=True,parents=True)
    runtime_dir = (output_dir / "runtime")
    runtime_dir.mkdir(exist_ok=True)
    performance_dir = (output_dir / "performance")
    performance_dir.mkdir(exist_ok=True)
    benchmark_files = [file for file in benchmark_dir.iterdir() if file.suffix == ".csv"]
    
    machine = get_machine_info()
    machine_frequency = frequency_to_number(machine.base_frequency)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot()

    for benchmark_file in benchmark_files:
        print(f"Reading {benchmark_file}")
        benchmark_df = pd.read_csv(benchmark_file)
        sanity_check_df(benchmark_df)
        benchmark_df = merge_unmerged(benchmark_df)
        experiment_names = get_experiment_names(benchmark_df["name"].unique())
        if benchmark_df.empty:
            if verbose:
                print(f"No data found for benchmark suite {benchmark_file.name}")
            continue
        conv_type_dir_runtime = runtime_dir / benchmark_file.name[:-4]
        conv_type_dir_runtime.mkdir(exist_ok=True)
        conv_type_dir_performance = performance_dir / benchmark_file.name[:-4]
        conv_type_dir_performance.mkdir(exist_ok=True)
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
            x_data = []
            performance_values = []
            runtime_values = []
            for exp_name in experiment_names:
                experiment_data = df_by_func[df_by_func["name"] == exp_name]
                if verbose:
                    print(f'Plotting data for Experiment {exp_name}, Benchmark suite {benchmark_file.name} and Function {function.value}')
                xs, ys_runtime, ys_performance = [], [], []
                for _, data_point in experiment_data.iterrows():
                    xs.append(get_input_size(data_point))
                    cost = Baseline(data_point).cost()
                    ys_runtime.append(data_point.cycles / machine_frequency)
                    ys_performance.append((cost.iops+cost.flops) / data_point.cycles)
                if len(xs) == 0:
                    continue
                xs, ys_runtime = unzip_data_points(xs,ys_runtime)

                x_data.append(xs)
                performance_values.append(ys_performance)
                runtime_values.append(ys_runtime)
            # Save to performance dir
            for exp_name,x,y in zip(experiment_names,x_data,performance_values):
                if len(x) < 2:
                    ax.scatter(x,y,label=experiment_names[exp_name])
                else:
                    ax.plot(x, y,label=experiment_names[exp_name])
            ax.set_xlabel("Input size")
            ax.set_ylabel('Performance [ops/cycle]',
                    rotation='horizontal',
                    loc='top',
                    labelpad=-112)
            set_plot_params(ax, machine, conv_type_dir_performance / function.value, function.fancy(), benchmark_file.name[:-4])
            # Save to runtime dir
            plt.clf()
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            ax = fig.add_subplot()
            for exp_name,x,y in zip(experiment_names,x_data,runtime_values):
                if len(xs) < 2:
                    ax.scatter(x,y,label=experiment_names[exp_name])
                else:
                    ax.plot(x, y,label=experiment_names[exp_name])
            ax.set_xlabel("Input size")
            ax.set_ylabel('Runtime [seconds]',
                    rotation='horizontal',
                    loc='top',
                    labelpad=-112)
            set_plot_params(ax, machine, conv_type_dir_runtime / function.value, function.fancy(), benchmark_file.name[:-4])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Plotter",
                                     description="Plot the benchmarks for ASL")
    parser.add_argument("-i","--input",default=str(DATA_DIR))
    parser.add_argument("-o","--output",default=str(PLOT_DIR))
    parser.add_argument('-v', '--verbose',
                    action='store_true')
    args = parser.parse_args()
    create_plots(Path(args.input),Path(args.output),args.verbose)
