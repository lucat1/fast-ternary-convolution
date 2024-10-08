"""Get machine info."""

# from help import run_cmd, output_to_dict
from dataclasses import dataclass
import cpuinfo

@dataclass
class MachInfo:
    """
    Machine info.

    This includes:

    - arcitecture: Yes.
    - model: The machine processor name.
    - base_frequency: In GHz.
    - operating_system: Yes.
    - L1_cache_size: In bytes.
    - L2_cache_size: In bytes.
    - L3_cache_size: In bytes.
    """
    architecture: str
    model: str
    base_frequency: float
    L1_cache_size: int
    L2_cache_size: int
    L3_cache_size: int


def get_machine_info() -> MachInfo:
    """Get machine information."""
    info = cpuinfo.get_cpu_info()
    architecture = info['arch']
    num_cores = info.get('count')
    model = info['brand_raw']
    
    L1_cache_size = info.get('l1_data_cache_size')
    L1_cache_size = L1_cache_size // num_cores if L1_cache_size is not None else L1_cache_size
    L2_cache_size = info.get('l2_cache_size')
    L2_cache_size = L2_cache_size // num_cores if L2_cache_size is not None else L2_cache_size
    L3_cache_size = info.get('l3_cache_size')
    base_frequency = float(info.get('hz_advertised_friendly').split()[0])
    return MachInfo(architecture, model, base_frequency, L1_cache_size, L2_cache_size, L3_cache_size)
