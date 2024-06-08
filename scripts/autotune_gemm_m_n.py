import subprocess
import time
import csv
from tqdm import tqdm

# Define the ranges for block sizes
n_m_block_sizes = [2 ** i for i in range(10)]  # Powers of two from 1 to 512
initial_k_block_sizes = [2 ** i for i in range(1,6)]  # Initial sequence for K_BLOCK_SIZE
additional_k_block_sizes = []


# Function to run a shell command and capture its output
def run_shell_command(command):
    process = subprocess.run(command, shell=True, capture_output=False)
    if process.returncode != 0:
        print(f"Command failed: {command}")
        print(process.stderr.decode())
    return process

# Function to build and run the project with given block sizes
def run_code_with_block_size(n_block_size, m_block_size, k_block_size):
    # Clean up build files
    run_shell_command('rm -rf build/impl/t2r_gemmLU_autoblock/')
    
    # Build the project with the given block sizes
    make_command = f'make EXTRA="-DN_BLOCK_SIZE={n_block_size} -DM_BLOCK_SIZE={m_block_size} -DK_BLOCK_SIZE={k_block_size}"'
    build_process = run_shell_command(make_command)
    if build_process.returncode != 0:
        return float('inf')
    
    # Run the benchmarks
    start_time = time.time()
    run_process = run_shell_command('taskset -a -c 0 ./tnn -b -p parameters/autotune.csv -o benchmarks/autotune.csv -i t2r_gemmLU_autoblock')
    end_time = time.time()
    
    if run_process.returncode != 0:
        return float('inf')
    
    runtime = end_time - start_time
    return runtime

# Dictionary to store runtimes for each block size
runtimes = {}
# Variables to store the best result found so far
current_optimal_block_size = None
current_optimal_runtime = float('inf')

# Calculate total iterations for tqdm
total_iterations = len(n_m_block_sizes) * len(n_m_block_sizes) * (len(initial_k_block_sizes) + len(additional_k_block_sizes))

# Open CSV file for writing
with open('benchmark_results.csv', mode='w', newline='') as csvfile:
    fieldnames = ['N_BLOCK_SIZE', 'M_BLOCK_SIZE', 'K_BLOCK_SIZE', 'Runtime']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Progress bar
    with tqdm(total=total_iterations, desc="Testing block sizes") as pbar:
        # Loop through each combination of block sizes, build, run, and measure the runtime
        for n_block_size in n_m_block_sizes:
            for m_block_size in n_m_block_sizes:
                # Track the number of consecutive steps without improvement
                no_improvement_steps = 0

                for k_block_size in initial_k_block_sizes:
                    print(f"Testing N_BLOCK_SIZE={n_block_size}, M_BLOCK_SIZE={m_block_size}, K_BLOCK_SIZE={k_block_size}")
                    runtime = run_code_with_block_size(n_block_size, m_block_size, k_block_size)
                    runtimes[(n_block_size, m_block_size, k_block_size)] = runtime
                    print(f"N_BLOCK_SIZE={n_block_size}, M_BLOCK_SIZE={m_block_size}, K_BLOCK_SIZE={k_block_size}, Runtime: {runtime:.4f} seconds")
                    pbar.update(1)
                    
                    # Write results to CSV
                    writer.writerow({'N_BLOCK_SIZE': n_block_size, 'M_BLOCK_SIZE': m_block_size, 'K_BLOCK_SIZE': k_block_size, 'Runtime': runtime})
                    csvfile.flush()

                    # Check if the current runtime is better than the current optimal runtime
                    if runtime < current_optimal_runtime:
                        current_optimal_runtime = runtime
                        current_optimal_block_size = (n_block_size, m_block_size, k_block_size)
                        print(f"New optimal found! N_BLOCK_SIZE={n_block_size}, M_BLOCK_SIZE={m_block_size}, K_BLOCK_SIZE={k_block_size} with runtime: {runtime:.4f} seconds")

                for k_block_size in additional_k_block_sizes:
                    print(f"Testing N_BLOCK_SIZE={n_block_size}, M_BLOCK_SIZE={m_block_size}, K_BLOCK_SIZE={k_block_size}")
                    runtime = run_code_with_block_size(n_block_size, m_block_size, k_block_size)
                    runtimes[(n_block_size, m_block_size, k_block_size)] = runtime
                    print(f"N_BLOCK_SIZE={n_block_size}, M_BLOCK_SIZE={m_block_size}, K_BLOCK_SIZE={k_block_size}, Runtime: {runtime:.4f} seconds")
                    pbar.update(1)
                        
                    # Write results to CSV
                    writer.writerow({'N_BLOCK_SIZE': n_block_size, 'M_BLOCK_SIZE': m_block_size, 'K_BLOCK_SIZE': k_block_size, 'Runtime': runtime})
                    csvfile.flush()

                    # Check if the current runtime is better than the current optimal runtime
                    if runtime < current_optimal_runtime:
                        current_optimal_runtime = runtime
                        current_optimal_block_size = (n_block_size, m_block_size, k_block_size)
                        print(f"New optimal found! N_BLOCK_SIZE={n_block_size}, M_BLOCK_SIZE={m_block_size}, K_BLOCK_SIZE={k_block_size} with runtime: {runtime:.4f} seconds")
                        no_improvement_steps = 0
                    else:
                        no_improvement_steps += 1
                    # If no improvement is found for two consecutive larger values, break early
                    if no_improvement_steps > 2:
                        break

# Print the final optimal block sizes and runtime
print(f"Optimal block sizes: N_BLOCK_SIZE={current_optimal_block_size[0]}, M_BLOCK_SIZE={current_optimal_block_size[1]}, K_BLOCK_SIZE={current_optimal_block_size[2]} with runtime: {current_optimal_runtime:.4f} seconds")
