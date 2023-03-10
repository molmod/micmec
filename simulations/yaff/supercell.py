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

# Credits to Juul S. De Vos for providing this script!

import os

import numpy as np

from yaff import System, log
log.set_level(0)
from yaff.pes.ext import Cell
from molmod.units import angstrom
from molmod.molecular_graphs import MolecularGraph


def get_cell_dimensions(n_elements):
    x_dim = 1
    y_dim = 1
    z_dim = 1
    for x, y, z in n_elements:
        if x + 1 > x_dim:
            x_dim = x + 1
        if y + 1 > y_dim:
            y_dim = y + 1
        if z + 1 > z_dim:
            z_dim = z + 1
    return x_dim, y_dim, z_dim


def create_supercell(supercell, sys, central_cut):
    # Create system matrix.
    reps = get_cell_dimensions(supercell.keys())
    super_sys = sys.supercell(*reps)
    graph = MolecularGraph(super_sys.bonds, super_sys.numbers)

    # Get the indices that are not allowed in the final structure.
    offset = 0
    indices_cut = []
    for iimage in np.ndindex(*reps):
        top = supercell[iimage]
        for center, border in central_cut[top]:
            # Border has to be defined as part of the cluster that has to be cut out.
            # All indices in the border have to be connected within a unit cell.
            new_center = center + offset
            new_border = [i + offset for i in border]
            indices_cut.extend(new_border)
            indices_cut.extend(graph.get_part(new_center, new_border))
        offset += sys.natom

    # Find the attributes of the final structure.
    final_numbers = []
    final_pos = []
    final_ffatypes = []
    final_bonds = []
    
    old_to_new = {} # Dict from indices in system matrix to indices in final structure
    for i in range(super_sys.natom):
        if i in indices_cut: continue # Indices that have to be cut out are not allowed
        old_to_new[i] = len(final_numbers)
        final_numbers.append(super_sys.numbers[i])
        final_pos.append(super_sys.pos[i])
        # Check if the atom has a neighbor that is cut out
        nei_out = None
        for j in super_sys.neighs1[i]:
            if j in indices_cut:
                nei_out = j
        if nei_out == None:
            final_ffatypes.append(super_sys.get_ffatype(i))
        else:
            # Atom is left dangling, cap with a formate hydrogen and change to a formate carbon.
            assert super_sys.get_ffatype(i) == "C_CA", "Atom {} is no C_CA, but {}".format(i, super_sys.get_ffatype(i))
            final_ffatypes.append("C_FO")
            axis = super_sys.pos[nei_out] - super_sys.pos[i]
            super_sys.cell.mic(axis) # Apply minimal image convention to point in correct direction.
            axis /= np.linalg.norm(axis)
            r0 = 1.1019507749*angstrom
            final_numbers.append(1)
            final_pos.append(super_sys.pos[i] + r0*axis)
            final_ffatypes.append("H_FO")
            final_bonds.append([len(final_numbers) - 2, len(final_numbers) - 1])

    # Add bonds.
    for i0, i1 in super_sys.bonds:
        if not (i0 in indices_cut or i1 in indices_cut):
            final_bonds.append([old_to_new[i0], old_to_new[i1]])

    final_numbers = np.array(final_numbers)
    final_pos = np.array(final_pos)
    final_ffatypes = np.array(final_ffatypes)
    final_bonds = np.array(final_bonds)
    final_rvecs = super_sys.cell.rvecs # Unit cell is same as system matrix.
    
    final = System(final_numbers, final_pos, ffatypes = final_ffatypes, bonds = final_bonds, rvecs = final_rvecs)

    return final


def check(supercell, system, n_fo = {"fcu": 0, "reo": 12}, n_bonds = {"fcu": 592, "reo": 384}):
    # Create check criteria.
    fo_groups = 0
    bonds = 0
    for iimage, top in supercell.items():
        fo_groups += n_fo[top]
        bonds += n_bonds[top]
    
    # Check 1: number of formate groups should be according to defect types.
    counter_h_fo = 0
    counter_c_fo = 0
    for i in range(system.natom):
        if system.get_ffatype(i) == "H_FO":
            counter_h_fo += 1
            for j in system.neighs1[i]:
                assert system.get_ffatype(j) == "C_FO" # Each formate hydrogen has a formate carbon.
        elif system.get_ffatype(i) == "C_FO":
            counter_c_fo += 1
    assert counter_h_fo == fo_groups and counter_c_fo == fo_groups
    
    # Check 2: number of bonds should be according to defect types.
    assert len(system.bonds) == bonds, "Expected {} bonds, got {}".format(bonds, len(system.bonds))

    # Check 3: maximum distance should be lower than 2.5A.
    max_dist = None
    for i, j in system.bonds:
        delta = system.pos[j] - system.pos[i]
        system.cell.mic(delta)
        dist = np.linalg.norm(delta)
        if max_dist == None or dist > max_dist:
            max_dist = dist
    assert max_dist < 2.5*angstrom


if __name__ == "__main__":
    # Input data: fcu unit cell and which atoms to cut away for each defect.
    central_cut = {
        "fcu": [],
        "reo": [[1, [239, 230, 220, 245, 216, 233, 227, 242, 209, 214, 207, 200]]],
    }
    central_sys = System.from_file("fcu.chk")
    
    # 3x3x3 structures to create:
    todo_3x3x3 = [
        ["conf0", 
            {
                (0,0,0): "fcu", #0
                (0,0,1): "fcu", #1
                (0,0,2): "fcu", #2
                (0,1,0): "fcu", #3
                (0,1,1): "fcu", #4
                (0,1,2): "fcu", #5
                (0,2,0): "fcu", #6
                (0,2,1): "fcu", #7
                (0,2,2): "fcu", #8
                (1,0,0): "fcu", #9
                (1,0,1): "fcu", #10
                (1,0,2): "fcu", #11
                (1,1,0): "fcu", #12
                (1,1,1): "fcu", #13
                (1,1,2): "fcu", #14
                (1,2,0): "fcu", #15
                (1,2,1): "fcu", #16
                (1,2,2): "fcu", #17
                (2,0,0): "fcu", #18
                (2,0,1): "fcu", #19
                (2,0,2): "fcu", #20
                (2,1,0): "fcu", #21
                (2,1,1): "fcu", #22
                (2,1,2): "fcu", #23
                (2,2,0): "fcu", #24
                (2,2,1): "fcu", #25
                (2,2,2): "fcu", #26
            }
        ],
        ["conf1", 
            {
                (0,0,0): "fcu", #0
                (0,0,1): "fcu", #1
                (0,0,2): "fcu", #2
                (0,1,0): "fcu", #3
                (0,1,1): "fcu", #4
                (0,1,2): "fcu", #5
                (0,2,0): "fcu", #6
                (0,2,1): "fcu", #7
                (0,2,2): "fcu", #8
                (1,0,0): "fcu", #9
                (1,0,1): "fcu", #10
                (1,0,2): "fcu", #11
                (1,1,0): "fcu", #12
                (1,1,1): "fcu", #13
                (1,1,2): "fcu", #14
                (1,2,0): "fcu", #15
                (1,2,1): "fcu", #16
                (1,2,2): "fcu", #17
                (2,0,0): "reo", #18 !
                (2,0,1): "fcu", #19
                (2,0,2): "fcu", #20
                (2,1,0): "fcu", #21
                (2,1,1): "fcu", #22
                (2,1,2): "fcu", #23
                (2,2,0): "fcu", #24
                (2,2,1): "fcu", #25
                (2,2,2): "fcu", #26
            }
        ],
        ["conf2", 
            {
                (0,0,0): "reo", #0 !
                (0,0,1): "fcu", #1
                (0,0,2): "fcu", #2
                (0,1,0): "fcu", #3
                (0,1,1): "fcu", #4
                (0,1,2): "fcu", #5
                (0,2,0): "fcu", #6
                (0,2,1): "fcu", #7
                (0,2,2): "fcu", #8
                (1,0,0): "reo", #9 !
                (1,0,1): "fcu", #10
                (1,0,2): "fcu", #11
                (1,1,0): "fcu", #12
                (1,1,1): "fcu", #13
                (1,1,2): "fcu", #14
                (1,2,0): "fcu", #15
                (1,2,1): "fcu", #16
                (1,2,2): "fcu", #17
                (2,0,0): "reo", #18 !
                (2,0,1): "fcu", #19
                (2,0,2): "fcu", #20
                (2,1,0): "fcu", #21
                (2,1,1): "fcu", #22
                (2,1,2): "fcu", #23
                (2,2,0): "fcu", #24
                (2,2,1): "fcu", #25
                (2,2,2): "fcu", #26
            }
        ],
        ["conf3",
            {
                (0,0,0): "fcu", #0
                (0,0,1): "fcu", #1
                (0,0,2): "fcu", #2
                (0,1,0): "fcu", #3
                (0,1,1): "fcu", #4
                (0,1,2): "fcu", #5
                (0,2,0): "reo", #6 !
                (0,2,1): "fcu", #7
                (0,2,2): "fcu", #8
                (1,0,0): "fcu", #9
                (1,0,1): "fcu", #10
                (1,0,2): "fcu", #11
                (1,1,0): "reo", #12 !
                (1,1,1): "fcu", #13
                (1,1,2): "fcu", #14
                (1,2,0): "fcu", #15
                (1,2,1): "fcu", #16
                (1,2,2): "fcu", #17
                (2,0,0): "reo", #18 !
                (2,0,1): "fcu", #19
                (2,0,2): "fcu", #20
                (2,1,0): "fcu", #21
                (2,1,1): "fcu", #22
                (2,1,2): "fcu", #23
                (2,2,0): "fcu", #24
                (2,2,1): "fcu", #25
                (2,2,2): "fcu", #26
            } 
        ],
        ["conf4", 
            {
                (0,0,0): "fcu", #0
                (0,0,1): "fcu", #1
                (0,0,2): "fcu", #2
                (0,1,0): "fcu", #3
                (0,1,1): "fcu", #4
                (0,1,2): "fcu", #5
                (0,2,0): "fcu", #6
                (0,2,1): "fcu", #7
                (0,2,2): "reo", #8 !
                (1,0,0): "fcu", #9
                (1,0,1): "fcu", #10
                (1,0,2): "fcu", #11
                (1,1,0): "fcu", #12
                (1,1,1): "reo", #13 !
                (1,1,2): "fcu", #14
                (1,2,0): "fcu", #15
                (1,2,1): "fcu", #16
                (1,2,2): "fcu", #17
                (2,0,0): "reo", #18 !
                (2,0,1): "fcu", #19
                (2,0,2): "fcu", #20
                (2,1,0): "fcu", #21
                (2,1,1): "fcu", #22
                (2,1,2): "fcu", #23
                (2,2,0): "fcu", #24
                (2,2,1): "fcu", #25
                (2,2,2): "fcu", #26
            }
        ],
        ["conf5", 
            {
                (0,0,0): "fcu", #0
                (0,0,1): "fcu", #1
                (0,0,2): "fcu", #2
                (0,1,0): "fcu", #3
                (0,1,1): "fcu", #4
                (0,1,2): "fcu", #5
                (0,2,0): "fcu", #6
                (0,2,1): "fcu", #7
                (0,2,2): "fcu", #8
                (1,0,0): "reo", #9 !
                (1,0,1): "reo", #10 !
                (1,0,2): "fcu", #11
                (1,1,0): "reo", #12 !
                (1,1,1): "reo", #13 !
                (1,1,2): "fcu", #14
                (1,2,0): "fcu", #15
                (1,2,1): "fcu", #16
                (1,2,2): "fcu", #17
                (2,0,0): "reo", #18 !
                (2,0,1): "reo", #19 !
                (2,0,2): "fcu", #20
                (2,1,0): "reo", #21 !
                (2,1,1): "reo", #22 !
                (2,1,2): "fcu", #23
                (2,2,0): "fcu", #24
                (2,2,1): "fcu", #25
                (2,2,2): "fcu", #26
            }
        ],
        ["conf6", 
            {
                (0,0,0): "reo", #0 !
                (0,0,1): "fcu", #1
                (0,0,2): "fcu", #2
                (0,1,0): "reo", #3 !
                (0,1,1): "fcu", #4
                (0,1,2): "fcu", #5
                (0,2,0): "reo", #6 !
                (0,2,1): "fcu", #7
                (0,2,2): "fcu", #8
                (1,0,0): "reo", #9 !
                (1,0,1): "fcu", #10
                (1,0,2): "fcu", #11
                (1,1,0): "reo", #12 !
                (1,1,1): "fcu", #13
                (1,1,2): "fcu", #14
                (1,2,0): "reo", #15 !
                (1,2,1): "fcu", #16
                (1,2,2): "fcu", #17
                (2,0,0): "reo", #18 !
                (2,0,1): "fcu", #19
                (2,0,2): "fcu", #20
                (2,1,0): "reo", #21 !
                (2,1,1): "fcu", #22
                (2,1,2): "fcu", #23
                (2,2,0): "reo", #24 !
                (2,2,1): "fcu", #25
                (2,2,2): "fcu", #26
            }
        ],
        ["conf7", 
            {
                (0,0,0): "reo", #0 !
                (0,0,1): "fcu", #1
                (0,0,2): "fcu", #2
                (0,1,0): "fcu", #3
                (0,1,1): "reo", #4 !
                (0,1,2): "fcu", #5 
                (0,2,0): "fcu", #6
                (0,2,1): "fcu", #7
                (0,2,2): "reo", #8 !
                (1,0,0): "reo", #9 !
                (1,0,1): "fcu", #10
                (1,0,2): "fcu", #11
                (1,1,0): "fcu", #12
                (1,1,1): "reo", #13 !
                (1,1,2): "fcu", #14
                (1,2,0): "fcu", #15
                (1,2,1): "fcu", #16
                (1,2,2): "reo", #17 !
                (2,0,0): "reo", #18 !
                (2,0,1): "fcu", #19
                (2,0,2): "fcu", #20
                (2,1,0): "fcu", #21 
                (2,1,1): "reo", #22 !
                (2,1,2): "fcu", #23
                (2,2,0): "fcu", #24
                (2,2,1): "fcu", #25
                (2,2,2): "reo", #26 !
            }
        ],
        ["conf8", 
            {
                (0,0,0): "fcu", #0
                (0,0,1): "fcu", #1
                (0,0,2): "reo", #2 !
                (0,1,0): "reo", #3 !
                (0,1,1): "fcu", #4
                (0,1,2): "fcu", #5
                (0,2,0): "fcu", #6
                (0,2,1): "reo", #7 !
                (0,2,2): "fcu", #8
                (1,0,0): "reo", #9 !
                (1,0,1): "fcu", #10
                (1,0,2): "fcu", #11
                (1,1,0): "fcu", #12
                (1,1,1): "reo", #13 !
                (1,1,2): "fcu", #14
                (1,2,0): "fcu", #15
                (1,2,1): "fcu", #16
                (1,2,2): "reo", #17 !
                (2,0,0): "fcu", #18
                (2,0,1): "reo", #19 !
                (2,0,2): "fcu", #20
                (2,1,0): "fcu", #21
                (2,1,1): "fcu", #22
                (2,1,2): "reo", #23 !
                (2,2,0): "reo", #24 !
                (2,2,1): "fcu", #25
                (2,2,2): "fcu", #26
            }
        ],
        ["conf9", 
            {
                (0,0,0): "reo", #0
                (0,0,1): "reo", #1
                (0,0,2): "reo", #2
                (0,1,0): "reo", #3
                (0,1,1): "reo", #4
                (0,1,2): "reo", #5
                (0,2,0): "reo", #6
                (0,2,1): "reo", #7
                (0,2,2): "reo", #8
                (1,0,0): "reo", #9
                (1,0,1): "reo", #10
                (1,0,2): "reo", #11
                (1,1,0): "reo", #12
                (1,1,1): "reo", #13
                (1,1,2): "reo", #14
                (1,2,0): "reo", #15
                (1,2,1): "reo", #16
                (1,2,2): "reo", #17
                (2,0,0): "reo", #18
                (2,0,1): "reo", #19
                (2,0,2): "reo", #20
                (2,1,0): "reo", #21
                (2,1,1): "reo", #22
                (2,1,2): "reo", #23
                (2,2,0): "reo", #24
                (2,2,1): "reo", #25
                (2,2,2): "reo", #26
            }        
        ]
    ]
    # 2x2x2 structures to create:
    todo_2x2x2 = [
        ["c0", 
            {
                (0,0,0): "fcu",
                (1,0,0): "fcu",
                (0,1,0): "fcu",
                (0,0,1): "fcu",
                (1,1,0): "fcu",
                (1,0,1): "fcu",
                (0,1,1): "fcu",
                (1,1,1): "fcu",
            }        
        ],
        ["c1", 
            {
                (0,0,0): "reo",
                (1,0,0): "fcu",
                (0,1,0): "fcu",
                (0,0,1): "fcu",
                (1,1,0): "fcu",
                (1,0,1): "fcu",
                (0,1,1): "fcu",
                (1,1,1): "fcu",
            }        
        ],
        ["c2", 
            {
                (0,0,0): "reo",
                (1,0,0): "reo",
                (0,1,0): "fcu",
                (0,0,1): "fcu",
                (1,1,0): "fcu",
                (1,0,1): "fcu",
                (0,1,1): "fcu",
                (1,1,1): "fcu",
            }        
        ],
        ["c3", 
            {
                (0,0,0): "reo",
                (1,0,0): "fcu",
                (0,1,0): "fcu",
                (0,0,1): "fcu",
                (1,1,0): "reo",
                (1,0,1): "fcu",
                (0,1,1): "fcu",
                (1,1,1): "fcu",
            }        
        ],
        ["c4", 
            {
                (0,0,0): "reo",
                (1,0,0): "fcu",
                (0,1,0): "fcu",
                (0,0,1): "fcu",
                (1,1,0): "fcu",
                (1,0,1): "fcu",
                (0,1,1): "fcu",
                (1,1,1): "reo",
            }        
        ],
        ["c5", 
            {
                (0,0,0): "reo",
                (1,0,0): "reo",
                (0,1,0): "reo",
                (0,0,1): "fcu",
                (1,1,0): "reo",
                (1,0,1): "fcu",
                (0,1,1): "fcu",
                (1,1,1): "fcu",
            }        
        ],
        ["c6", 
            {
                (0,0,0): "reo",
                (1,0,0): "reo",
                (0,1,0): "fcu",
                (0,0,1): "fcu",
                (1,1,0): "fcu",
                (1,0,1): "fcu",
                (0,1,1): "reo",
                (1,1,1): "reo",
            }        
        ],
        ["c9", 
            {
                (0,0,0): "reo",
                (1,0,0): "reo",
                (0,1,0): "reo",
                (0,0,1): "reo",
                (1,1,0): "reo",
                (1,0,1): "reo",
                (0,1,1): "fcu",
                (1,1,1): "fcu",
            }        
        ],
        ["c8", 
            {
                (0,0,0): "reo",
                (1,0,0): "reo",
                (0,1,0): "reo",
                (0,0,1): "fcu",
                (1,1,0): "reo",
                (1,0,1): "reo",
                (0,1,1): "reo",
                (1,1,1): "fcu",
            }        
        ],
        ["c7", 
            {
                (0,0,0): "fcu",
                (1,0,0): "reo",
                (0,1,0): "reo",
                (0,0,1): "reo",
                (1,1,0): "reo",
                (1,0,1): "reo",
                (0,1,1): "reo",
                (1,1,1): "fcu",
            }        
        ],
        ["c10", 
            {
                (0,0,0): "reo",
                (1,0,0): "reo",
                (0,1,0): "reo",
                (0,0,1): "reo",
                (1,1,0): "reo",
                (1,0,1): "reo",
                (0,1,1): "reo",
                (1,1,1): "fcu",
            }        
        ],
        ["c11", 
            {
                (0,0,0): "reo",
                (1,0,0): "reo",
                (0,1,0): "reo",
                (0,0,1): "reo",
                (1,1,0): "reo",
                (1,0,1): "reo",
                (0,1,1): "reo",
                (1,1,1): "reo",
            }        
        ]
    ]
    
    todo = todo_3x3x3 + todo_2x2x2

    for name, supercell in todo:
        print(name)
        sys = create_supercell(supercell, central_sys, central_cut)
        try:
            check(supercell, sys)
        except Exception as e:
            print(e)
        sys.to_file("{}_atomic.chk".format(name))


