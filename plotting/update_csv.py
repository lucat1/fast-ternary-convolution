"""
Update one csv file with implementation data from another csv.

The old data is replaced and wiped entirely.
"""

import argparse
import pandas as pd


def transfer_data(source_file, target_file, implementation_name):
    """Replace data for implementation_name in target_file with that from source_file."""
    # Load the source and target CSV files
    source_data = pd.read_csv(source_file)
    target_data = pd.read_csv(target_file)

    # Filter data based on the implementation name from the source data
    filtered_data = source_data[source_data['name'] == implementation_name]

    # Remove any existing rows in the target data that have the same implementation name
    target_data = target_data[target_data['name'] != implementation_name]

    # Append the filtered data from the source to the target data
    updated_data = pd.concat([target_data, filtered_data])

    # Save the updated data back to the target CSV file
    updated_data.to_csv(target_file, index=False)
    print(f"Data for implementation '{implementation_name}' transferred and target CSV updated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer data between CSV files based on implementation name.')
    parser.add_argument('-s', '--source', type=str, required=True, help='Source CSV file')
    parser.add_argument('-t', '--target', type=str, required=True, help='Target CSV file')
    parser.add_argument('-n', '--name', type=str, required=True, help='Implementation name to transfer')

    args = parser.parse_args()

    # Call the data transfer function
    transfer_data(args.source, args.target, args.name)
