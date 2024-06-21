"""Calculate speed up between 2 implementations."""

import pandas as pd
import argparse


def get_speedup(data_file, func, impl1, impl2):
    """Get speedup info for data in file."""
    data = pd.read_csv(data_file)

    data = data[data['fn'] == func]
    
    impl1_data = data[data['name'] == impl1]
    impl2_data = data[data['name'] == impl2]

    for (_, i1), (_, i2) in zip(impl1_data.iterrows(), impl2_data.iterrows()):
        print(f'{impl2} clocked at {i2.cycles} is {i1.cycles / i2.cycles} faster than {impl1} clocked at {i1.cycles}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='GetSpeedUp',
        description='Get speedup between 2 implementations')
    parser.add_argument('-d', '--data',
                        type=argparse.FileType('r'),
                        required=True, help='CSV file with measurement data.')
    parser.add_argument('-f', '--function',
                        type=str,
                        required=True, help='function to consider.')
    parser.add_argument('-i1', '--impl1',
                        type=str, required=True,
                        help='implementation 1')
    parser.add_argument('-i2', '--impl2',
                        type=str, required=True,
                        help='implementation 2')

    args = parser.parse_args()

    get_speedup(args.data, args.function, args.impl1, args.impl2)
