#!/usr/bin/env python
# File name: trajectory.py
# Description: Trajectory writer for the micromechanical model.
# Author: Joachim Vandewalle
# Date: 18-11-2021

'''Trajectory writers'''

from iterative import Hook
from log import log, timer

__all__ = ['HDF5Writer', 'XYZWriter']


class BaseHDF5Writer(Hook):

    def __init__(self, f, start=0, step=1):
        """
        **Argument:**
        f
            A h5.File object to write the trajectory to.
        
        **Optional arguments:**
        start
            The first iteration at which this hook should be called.
        step
            The hook will be called every `step` iterations.
        """
        self.f = f
        Hook.__init__(self, start, step)

    
    def __call__(self, iterative):
        if 'trajectory' not in self.f:
            self.init_trajectory(iterative)
        tgrp = self.f['trajectory']
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
                ds.resize(row+1, axis=0)
            ds[row] = item.value
    
    def dump_system(self, system, grp):
        system.to_hdf5(grp)

    def init_trajectory(self, iterative):
        tgrp = self.f.create_group('trajectory')
        for key, item in iterative.state.items():
            if len(item.shape) > 0 and min(item.shape) == 0:
                continue
            if item.value is None:
                continue
            maxshape = (None,) + item.shape
            shape = (0,) + item.shape
            dset = tgrp.create_dataset(key, shape, maxshape=maxshape, dtype=item.dtype)
            for name, value in item.iter_attrs(iterative):
                tgrp.attrs[name] = value


class HDF5Writer(BaseHDF5Writer):
    
    def __call__(self, iterative):
        if 'system' not in self.f:
            self.dump_system(iterative.mmf.system, self.f)
        BaseHDF5Writer.__call__(self, iterative)



class XYZWriter(Hook):

    def __init__(self, fn_xyz, select=None, start=0, step=1):
        """
        **Argument:**
        fn_xyz
            A filename to write the XYZ trajectory to.

        **Optional arguments:**
        select
            A list of node indexes that should be written to the trajectory
            output. If not given, all nodes are included.
        start
            The first iteration at which this hook should be called.
        step
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
                symbols = [periodic[55].symbol for _ in pos] # represent nodes with cesium atoms
            else:
                symbols = [periodic[55].symbol for _ in self.select] # represent nodes with cesium atoms
            self.xyz_writer = XYZWriter(self.fn_xyz, symbols)
        title = '%7i E_pot = %.10f     ' % (iterative.counter, iterative.epot)
        if self.select is None:
            pos = iterative.mmf.system.pos
        else:
            pos = iterative.mmf.system.pos[self.select]
        self.xyz_writer.dump(title, pos)



