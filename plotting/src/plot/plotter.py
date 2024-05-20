"""Makes a plot from CSV."""

import matplotlib.pyplot as plt
from plot.datatypes import ConvType,Function
from plot.opcount.util import get_work_for_function, get_data_movement_for_function
from plot.utils import set_plot_params,unzip_data_points,get_batch_size,get_input_size
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
plt.rcParams['legend.fontsize'] = 11
#plt.rcParams['font.family'] = 'New Computer Modern'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
plt.rcParams['svg.fonttype'] = 'none'


conv_types = [conv_type for conv_type in ConvType]
functions = [function_type for function_type in Function]
csv_columns = ["name","ct","fn","cycles","channels",
               "batch_size","kernel_number","input_height","input_width",
               "kernel_height","kernel_width","padding_size","stride_size"]

conv_types_to_functions = {
    ConvType.TNN : [
        Function.TERNARIZE,Function.IMG2ROW,
        Function.TNN_GEMM,Function.ALLOC,Function.ALLOC2,
        Function.FREE,Function.PRELU,Function.CONV
    ],
    ConvType.TBN: [
        Function.TERNARIZE,Function.IMG2ROW,
        Function.TBN_GEMM,Function.ALLOC,Function.ALLOC2,
        Function.FREE,Function.PRELU,Function.CONV
    ],
    ConvType.BNN: [
        Function.BINARIZE,Function.IMG2ROW,
        Function.BNN_GEMM,Function.ALLOC,Function.ALLOC2,
        Function.FREE,Function.PRELU,Function.CONV
    ],
    ConvType.BTN: [
        Function.BINARIZE,Function.IMG2ROW,
        Function.BTN_GEMM,Function.ALLOC,Function.ALLOC2,
        Function.FREE,Function.PRELU,Function.CONV
    ]
}

def sanity_check_df(benchmark_df: pd.DataFrame):
    df_column_set = set(benchmark_df.columns)
    correct_columns_set = set(csv_columns)
    assert df_column_set == correct_columns_set, f"The csv does not have the correct columns. Columns should be {csv_columns}"


def create_plots(benchmark_dir: Path, output_dir: Path,verbose:bool) -> None:
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
        functions = [Function[func.upper()] for func in benchmark_df["fn"].unique()]
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
                    xs.append(get_input_size(data_point))
                    # xs.append(get_input_size(data_point))
                    iops, flops = get_work_for_function(data_point)
                    # print(iops, flops)
                    if function == Function.TERNARIZE:
                        data_movement = get_data_movement_for_function(data_point)
                        if verbose:
                            print(f"{function.value} I = {iops+flops}/{data_movement} = {(iops+flops)/data_movement}")
                    ys.append((iops+flops) / data_point.cycles)
                xs, ys = unzip_data_points(xs,ys)
                if len(xs) < 2:
                    ax.scatter(xs,ys,label=exp_name)
                else:
                    ax.plot(xs, ys,label=exp_name)
            set_plot_params(ax, machine, conv_type_dir / function.value, function.fancy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Plotter",
                                     description="Plot the benchmarks for ASL")
    parser.add_argument("-i","--input",default=str(DATA_DIR))
    parser.add_argument("-o","--output",default=str(PLOT_DIR))
    parser.add_argument('-v', '--verbose',
                    action='store_true')
    args = parser.parse_args()
    print(args)
    create_plots(Path(args.input),Path(args.output),args.verbose)