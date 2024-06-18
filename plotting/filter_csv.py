"""Filter implementations."""

import pandas as pd
import argparse


def filter_implementations(input_file, output_file, implementations):
    """Filter implementations from an input csv input_file."""
    # Load the CSV file
    data = pd.read_csv(input_file)

    # Filter data based on implementations provided
    filtered_data = data[data['name'].isin(implementations)]

    # Save the filtered data to a new CSV file
    filtered_data.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='FilterImpl',
        description='Filter csvs by implementation')
    parser.add_argument('-i', '--input',
                        type=argparse.FileType('r'),
                        required=True, help='Input CSV file')
    parser.add_argument('-o', '--output',
                        type=argparse.FileType('w'), required=True,
                        help='Output CSV file')
    parser.add_argument('-n', '--names',
                        nargs='+', required=True,
                        help='List of implementation names to filter')

    args = parser.parse_args()
    input_csv = args.input
    output_csv = args.output
    implementations = args.names

    # Call the filtering function
    filter_implementations(input_csv, output_csv, implementations)
