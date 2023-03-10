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


"""Trajectory writers."""

from micmec.sampling.iterative import Hook

__all__ = [
    "HDF5Writer",
    "XYZWriter",
]


class BaseHDF5Writer(Hook):
    def __init__(self, f, start=0, step=1):
        """
        Parameters
        ----------
        f : h5py.File object (open)
            An .h5 file to write the trajectory to.
        start : int, optional
            The first iteration at which this hook should be called.
        step : int, optional
            The hook will be called every `step` iterations.
        """
        self.f = f
        Hook.__init__(self, start, step)

    def __call__(self, iterative):
        if "trajectory" not in self.f:
            self.init_trajectory(iterative)
        tgrp = self.f["trajectory"]
        # Determine the row to write the current iteration to. If a previous
        # iterations was not completely written, then the last row is reused.
        row = min(tgrp[key].shape[0] for key in iterative.state if key in tgrp.keys())
        for key, item in iterative.state.items():
            if item.value is None:
                continue
            if len(item.shape) > 0 and min(item.shape) == 0:
                continue
            ds = tgrp[key]
            if ds.shape[0] <= row:
                # Do not over-allocate; hdf5 works with chunks internally.
                ds.resize(row + 1, axis=0)
            ds[row] = item.value

    def dump_system(self, system, grp):
        system.to_hdf5(grp)

    def init_trajectory(self, iterative):
        tgrp = self.f.create_group("trajectory")
        for key, item in iterative.state.items():
            if len(item.shape) > 0 and min(item.shape) == 0:
                continue
            if item.value is None:
                continue
            maxshape = (None,) + item.shape
            shape = (0,) + item.shape
            _ = tgrp.create_dataset(key, shape, maxshape=maxshape, dtype=item.dtype)
            for name, value in item.iter_attrs(iterative):
                tgrp.attrs[name] = value


class HDF5Writer(BaseHDF5Writer):
    def __call__(self, iterative):
        if "system" not in self.f:
            self.dump_system(iterative.mmf.system, self.f)
        BaseHDF5Writer.__call__(self, iterative)


class XYZWriter(Hook):
    def __init__(self, fn_xyz, select=None, start=0, step=1):
        """
        Parameters
        ----------
        fn_xyz : str
            A filename to write the XYZ trajectory to.
        select : list, optional
            A selection of nodes whose degrees of freedom are included.
            If no list is given, all nodal coordinates are included.
        start : int, optional
            The first iteration at which this hook should be called.
        step : int, optional
            The hook will be called every `step` iterations.
        """
        self.fn_xyz = fn_xyz
        self.select = select
        self.xyz_writer = None
        Hook.__init__(self, start, step)

    def __call__(self, iterative):
        from molmod.periodic import periodic
        from molmod.io import XYZWriter

        if self.xyz_writer is None:
            pos = iterative.mmf.system.pos
            if self.select is None:
                symbols = [
                    periodic[55].symbol for _ in pos
                ]  # represent nodes with cesium atoms
            else:
                symbols = [
                    periodic[55].symbol for _ in self.select
                ]  # represent nodes with cesium atoms
            self.xyz_writer = XYZWriter(self.fn_xyz, symbols)
        title = "%7i E_pot = %.10f     " % (iterative.counter, iterative.epot)
        if self.select is None:
            pos = iterative.mmf.system.pos
        else:
            pos = iterative.mmf.system.pos[self.select]
        self.xyz_writer.dump(title, pos)
