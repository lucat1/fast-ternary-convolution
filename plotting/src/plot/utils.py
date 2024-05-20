"""Helpers for plotting utilities."""

import subprocess
import matplotlib.pyplot as plt
from plot.datatypes import Function
from plot.machine_info import MachInfo
import pandas as pd
from pathlib import Path

POPCNT_OPS = 1
CNTBITS = 64
BITS = 2

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
    return (benchmark_info.batch_size * benchmark_info.input_height*benchmark_info.input_width * benchmark_info.channels)

def get_batch_size(benchmark_info: pd.Series) -> int:
    return benchmark_info.batch_size


def set_plot_params(ax: plt.Axes, machine: MachInfo, sav_loc: Path, function: Function):
    # x_label="input_size"
    ax.set_xlabel("Input size")
    ax.set_ylabel('Performance [ops/cycle]',
            rotation='horizontal',
            loc='top',
            labelpad=-112)
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

    title = f"{machine.model} @ {machine.base_frequency}"
    ax.set_title(title)
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
