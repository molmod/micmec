#!/usr/bin/env python
# File name: iterative.py
# Description: Iterative object for use during simulation.
# Author: Joachim Vandewalle
# Date: 18-11-2021

"""Base class for iterative algorithms."""

import numpy as np

from ..log import log, timer

from molmod.units import *


__all__ = ["Iterative", "StateItem", "AttributeStateItem", "PosStateItem", "EPotContribStateItem", "ConsErrStateItem",
            "TemperatureStateItem", "VolumeStateItem", "DomainStateItem", "Hook"]

class Iterative(object):
    
    default_state = []
    log_name = "ITER"

    def __init__(self, mmf, state=None, hooks=None, counter0=0):
        
        self.mmf = mmf
        
        if state is None:
            self.state_list = [state_item.copy() for state_item in self.default_state]
        else:
            #self.state_list = state
            self.state_list = [state_item.copy() for state_item in self.default_state]
            self.state_list += state
        self.state = dict((item.key, item) for item in self.state_list)

        if hooks is None:
            self.hooks = []
        elif hasattr(hooks, "__len__"):
            self.hooks = hooks
        else:
            self.hooks = [hooks]
        self._add_default_hooks()

        self.counter0 = counter0
        self.counter = counter0
        
        with log.section(self.log_name), timer.section(self.log_name):
            self.initialize()


    def _add_default_hooks(self):
        pass

    def initialize(self):
        self.call_hooks()

    
    def call_hooks(self):
        with timer.section("%s hooks" % self.log_name):
            state_updated = False
            for hook in self.hooks:
                if hook.expects_call(self.counter):
                    if not state_updated:
                        for item in self.state_list:
                            item.update(self)
                        state_updated = True
                    hook(self)

    
    def run(self, nsteps=None):
        with log.section(self.log_name), timer.section(self.log_name):
            if nsteps is None:
                while True:
                    if self.propagate():
                        break
            else:
                for i in range(nsteps):
                    if self.propagate():
                        break
            self.finalize()

    def propagate(self):
        self.counter += 1
        self.call_hooks()

    def finalize():
        raise NotImplementedError


class StateItem(object):
    
    def __init__(self, key):
        self.key = key
        self.shape = None
        self.dtype = None

    def update(self, iterative):
        self.value = self.get_value(iterative)
        if self.shape is None:
            if isinstance(self.value, np.ndarray):
                self.shape = self.value.shape
                self.dtype = self.value.dtype
            else:
                self.shape = tuple([])
                self.dtype = type(self.value)

    def get_value(self, iterative):
        raise NotImplementedError

    def iter_attrs(self, iterative):
        return []

    def copy(self):
        return self.__class__()


class AttributeStateItem(StateItem):
    def get_value(self, iterative):
        return getattr(iterative, self.key, None)

    def copy(self):
        return self.__class__(self.key)


class PosStateItem(StateItem):  
    
    def __init__(self):
        StateItem.__init__(self, "pos")

    def get_value(self, iterative):
        return iterative.mmf.system.pos


class TemperatureStateItem(StateItem):
    
    def __init__(self):
        StateItem.__init__(self, "temp")

    def get_value(self, iterative):
        return getattr(iterative, "temp", None)

    def iter_attrs(self, iterative):
        yield "ndof", iterative.ndof


class VolumeStateItem(StateItem):
    def __init__(self):
        StateItem.__init__(self, "volume")

    def get_value(self, iterative):
        return iterative.mmf.system.domain.volume


class DomainStateItem(StateItem):
    def __init__(self):
        StateItem.__init__(self, "domain")

    def get_value(self, iterative):
        return iterative.mmf.system.domain.rvecs


class ConsErrStateItem(StateItem):
    def get_value(self, iterative):
        return getattr(iterative._cons_err_tracker, self.key, None)

    def copy(self):
        return self.__class__(self.key)


class EPotContribStateItem(StateItem):
    """Keeps track of all the contributions to the potential energy."""
    def __init__(self):
        StateItem.__init__(self, "epot_contribs")

    def get_value(self, iterative):
        return np.array([part.energy for part in iterative.mmf.parts])

    def iter_attrs(self, iterative):
        yield "epot_contrib_names", np.array([part.name for part in iterative.mmf.parts], dtype="S")


class Hook(object):
    name = None
    kind = None
    method = None
    def __init__(self, start=0, step=1):
        """
        Parameters
        ----------
        start : int
            The first iteration at which this hook should be called.
        step : int
            The hook will be called every `step` iterations.
        
        """
        self.start = start
        self.step = step

    def expects_call(self, counter):
        return counter >= self.start and (counter - self.start) % self.step == 0

    def __call__(self, iterative):
        raise NotImplementedError


