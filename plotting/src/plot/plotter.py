"""Makes a plot from CSV."""

import matplotlib.pyplot as plt
from plot.datatypes import ConvType,Function
from plot.opcount.util import get_work_for_function, get_data_movement_for_function
from plot.utils import set_plot_params,unzip_data_points,get_batch_size,get_input_size
from plot.machine_info import get_machine_info
from pathlib import Path
import pandas as pd

REPO_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = REPO_DIR / "benchmarks"
PLOT_DIR = REPO_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True,parents=True)

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

benchmark_csv = DATA_DIR / "batch_size_xl.csv"
# benchmark_csv = DATA_DIR / "input_size_md.csv"
# benchmark_csv = DATA_DIR / "test.csv"

benchmark_df = pd.read_csv(benchmark_csv)

machine = get_machine_info()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()

for conv_type in conv_types:
    conv_type_dir = PLOT_DIR / conv_type.value
    conv_type_dir.mkdir(exist_ok=True)
    df_conv_type = benchmark_df[benchmark_df["ct"] == conv_type.value]
    for function in conv_types_to_functions[conv_type]:
        plt.clf()
        ax = fig.add_subplot()
        print(f'Plotting data for ConvType {conv_type.value} and Function {function.value}')
        df_by_func = df_conv_type[df_conv_type["fn"] == function.value]
        xs, ys = [], []
        for _,data_point in df_by_func.iterrows():
            # print(data_point)
            xs.append(get_batch_size(data_point))
            # xs.append(get_input_size(data_point))
            iops, flops = get_work_for_function(data_point)
            # print(iops, flops)
            if function == Function.TERNARIZE:
                data_movement = get_data_movement_for_function(data_point)
                print(f"{function.value} I = {iops+flops}/{data_movement} = {(iops+flops)/data_movement}")
            ys.append((iops+flops) / data_point.cycles)
        xs, ys = unzip_data_points(xs,ys)
        ax.plot(xs, ys)
        set_plot_params(ax, machine, conv_type_dir / function.value, function.fancy())
