"""Helpers for plotting utilities."""

import subprocess
import matplotlib.pyplot as plt
from plot.datatypes import Function
from plot.machine_info import MachInfo
import pandas as pd
from pathlib import Path

POPCNT_OPS = 3
CNTBITS = 64
BITS = 2

EXPERIMENT_NAMES = {
    # "optmerge_im2row_ternarize": "Merge im2row+ternarize (optimized)",
    # "merge_im2row_ternarize": "Merge im2row+ternarize",
    # "indirect_nhwc": "Indirect convolutions (NHWC)",
    # "more_indirect_prelu_nhwc": "More Indirect convolutions + PReLU (NHWC)",
    # "more_indirect_nhwc": "More Indirect Convolutions (NHWC)",
    # "indirect_nhwc": "Indirect Convolutions (NHWC)",
    # "baseline_original": "Baseline (original)",
    # "nhwc": "Baseline (NHWC)",
    # "ternary_nhwc": "Ternary operators (NHWC)",
    # "nchw": "Baseline (NCHW)"
}

def run_cmd(cmd: list[str]) -> str:
    """Run a given command and get output."""
    return subprocess.check_output(cmd).decode("utf-8")


def output_to_dict(cmd_output: str) -> dict[str, str]:
    """Gets a dictionary of the output of a command."""
    output_dict = {}
    for line in cmd_output.split('\n'):
        if line:  # Check if line is not empty
            key, value = line.split(':', 1)  # Split at the first colon
            output_dict[key.strip()] = value.strip()
    return output_dict

def get_input_size(benchmark_info: pd.Series) -> int:
    # return (
    #     8*benchmark_info.channels*benchmark_info.batch_size
    #         *benchmark_info.input_height*benchmark_info.input_width
    # )
    return (benchmark_info.batch_size * benchmark_info.input_height*benchmark_info.input_width * benchmark_info.channels * benchmark_info.kernel_width * benchmark_info.kernel_height)

def get_batch_size(benchmark_info: pd.Series) -> int:
    return benchmark_info.batch_size


def set_plot_params(ax: plt.Axes, machine: MachInfo, sav_loc: Path, function: Function, file: str):
    # ax.legend(loc='upper center',
    #         bbox_to_anchor=(0.5, 1.17),
    #         ncol=15,
    #         borderpad=.5)

    ax.set_facecolor((0.9, 0.9, 0.9))
    ax.tick_params(axis='both', which='major', pad=15)
    ax.grid(which='major', axis='y', linewidth=2, color='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xscale("log")
    #ax.set_yscale("log")

    title = f"{machine.model}"
    ax.set_title(f"{title} ({file})")
    ax.legend()
    plt.suptitle(function)
    
    plt.savefig(sav_loc)

def unzip_data_points(data_x: list[int], data_y: list[int]) -> tuple[list[int],list[int]]:
    """Unzip data points."""
    unzipped = list(zip(*sorted(zip(data_x, data_y), key=lambda a: a[0])))
    input_sizes, flops_per_cycles = unzipped
    return list(input_sizes), list(flops_per_cycles)


def get_cache_boundary(cache_size: int) -> float:
    """Get cache boundary for plot."""
    # TODO
    return 0

def get_experiment_masks(df:pd.DataFrame)-> list[tuple[pd.DataFrame,str]]:
    masks = []
    masks.append(((df["kernel_height"] == 3) & (df["kernel_width"] == 3) 
                 & (df["stride_size"] == 1) & (df["input_height"] <= 56)
                 & (df["channels"] <=256),"Varying tensor shape")
    )
    masks.append((df["channels"] == 80,"Varying conv stride"))
    masks.append((df["channels"] == 512,"Varying kernel size"))
    #TODO: Will add this mask later
    #masks.append((df["input_height"] == 1,"Varying kernel number"))
    return masks

def get_experiment_names(experiments: pd.Series) -> dict[str,str]:
    mapping = {}
    for elem in experiments:
        if elem in EXPERIMENT_NAMES:
            mapping[elem] = EXPERIMENT_NAMES[elem]
        else:
            mapping[elem] = elem
    return mapping


def frequency_to_number(frequency: str) -> float:
    number, unit = frequency.split()
    number = float(number)
    if unit == "GHz":
        return number * 10**9
    if unit == "MHz":
        return number * 10**6
    raise ValueError(f"Invalid frequency given: {frequency}")
