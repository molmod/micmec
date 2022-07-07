#!/usr/bin/env python
# File name: add_numbers.py
# Description: Add an atomic numbers group to a .h5 file (via console).
# Author: Joachim Vandewalle
# Date: 18-03-2022

import numpy as np
import h5py

def main(h5_fn, num):
    with h5py.File(h5_fn, mode = 'a') as f:
        atomic_number = num
        num_nodes = len(np.array(f['system/pos']))
        f['system/numbers'] = atomic_number*np.ones((num_nodes,))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a fictitious atomic numbers group to an existing .h5 trajectory.")
    parser.add_argument("input_fn", type=str,
                        help=".h5 filename of the input trajectory")
    parser.add_argument("-num", type=int, default=55,
                        help="atomic number to represent the micromechanical nodes")

    args = parser.parse_args()
    main(args.input_fn, args.num)
