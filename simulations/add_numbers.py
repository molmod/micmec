#!/usr/bin/env python

#   MicMec 1.0, the first implementation of the micromechanical model, ever.
#               Copyright (C) 2022  Joachim Vandewalle
#                    joachim.vandewalle@hotmail.be
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, either version 3 of the License, or
#                  (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#              GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see https://www.gnu.org/licenses/.

"""Small convenience script to be able to view micromechanical trajectories.

It adds fake atomic numbers to the micromechanical nodes.
"""

import numpy as np
import h5py
import argparse


def main(h5_fn, num):
    with h5py.File(h5_fn, mode = 'a') as f:
        atomic_number = num
        num_nodes = len(np.array(f['system/pos']))
        f['system/numbers'] = atomic_number * np.ones((num_nodes,))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a fictitious atomic numbers group to an existing .h5 trajectory.")
    parser.add_argument("input_fn", type=str,
                        help=".h5 filename of the input trajectory")
    parser.add_argument("-num", type=int, default=55,
                        help="atomic number to represent the micromechanical nodes")

    args = parser.parse_args()
    main(args.input_fn, args.num)
