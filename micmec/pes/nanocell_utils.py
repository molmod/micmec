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


"""Fixed parameters for the evaluation of a nanocell's potential energy surface. DO NOT CHANGE."""

import numpy as np

from micmec.utils import neighbor_cells

__all__ = ["multiplicator", "cell_xderivs", "cell_yderivs", "cell_zderivs"]


# Construct a multiplicator array.
# This array converts the eight Cartesian coordinate vectors of a cell's surrounding nodes into eight matrix representations.
multiplicator = np.array(
    [
        [
            [-1, 1, 0, 0, 0, 0, 0, 0],
            [-1, 0, 1, 0, 0, 0, 0, 0],
            [-1, 0, 0, 1, 0, 0, 0, 0],
        ],
        [
            [-1, 1, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 1, 0, 0, 0],
            [0, -1, 0, 0, 0, 1, 0, 0],
        ],
        [
            [0, 0, -1, 0, 1, 0, 0, 0],
            [-1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 1, 0],
        ],
        [
            [0, 0, 0, -1, 0, 1, 0, 0],
            [0, 0, 0, -1, 0, 0, 1, 0],
            [-1, 0, 0, 1, 0, 0, 0, 0],
        ],
        [
            [0, 0, -1, 0, 1, 0, 0, 0],
            [0, -1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 1],
        ],
        [
            [0, 0, 0, -1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 1],
            [0, -1, 0, 0, 0, 1, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, -1, 1],
            [0, 0, 0, -1, 0, 0, 1, 0],
            [0, 0, -1, 0, 0, 0, 1, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, -1, 1],
            [0, 0, 0, 0, 0, -1, 0, 1],
            [0, 0, 0, 0, -1, 0, 0, 1],
        ],
    ]
)

# Initialize the derivatives of the neighboring cell matrices to x, y and z.
cell_xderivs = []
cell_yderivs = []
cell_zderivs = []
for neighbor_cell in neighbor_cells:
    xderivs = []
    yderivs = []
    zderivs = []
    for cell_representation in neighbor_cells:
        # (3.20) and (3.32)
        xderiv = np.zeros((3, 3))
        yderiv = np.zeros((3, 3))
        zderiv = np.zeros((3, 3))
        deriv = np.array([1.0 if n == -1 else -1.0 for n in neighbor_cell])
        dist_vec = np.abs(
            [e1 - e2 for e1, e2 in zip(neighbor_cell, cell_representation)]
        )
        dist = np.sum(dist_vec)
        if dist == 0.0:
            xderiv[:, 0] = deriv
            yderiv[:, 1] = deriv
            zderiv[:, 2] = deriv
        elif dist == 1.0:
            xderiv[dist_vec == 1.0, 0] = deriv[dist_vec == 1.0]
            yderiv[dist_vec == 1.0, 1] = deriv[dist_vec == 1.0]
            zderiv[dist_vec == 1.0, 2] = deriv[dist_vec == 1.0]
        else:
            pass
        xderivs.append(xderiv.T)
        yderivs.append(yderiv.T)
        zderivs.append(zderiv.T)
    cell_xderivs.append(xderivs)
    cell_yderivs.append(yderivs)
    cell_zderivs.append(zderivs)
