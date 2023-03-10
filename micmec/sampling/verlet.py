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


"""The velocity Verlet algorithm."""

import time
import numpy as np

from molmod import boltzmann, kelvin

from micmec.sampling.iterative import (
    Iterative,
    Hook,
    AttributeStateItem,
    PosStateItem,
    TemperatureStateItem,
    VolumeStateItem,
    DomainStateItem,
)
from micmec.sampling.utils import get_random_vel, clean_momenta
from micmec.log import log, timer


__all__ = [
    "VerletIntegrator",
    "VerletHook",
    "VerletScreenLog",
    "ConsErrTracker",
]


class VerletIntegrator(Iterative):
    default_state = [
        AttributeStateItem("counter"),
        AttributeStateItem("time"),
        AttributeStateItem("epot"),
        PosStateItem(),
        AttributeStateItem("vel"),
        AttributeStateItem("rmsd_delta"),
        AttributeStateItem("rmsd_gpos"),
        AttributeStateItem("ekin"),
        TemperatureStateItem(),
        AttributeStateItem("etot"),
        AttributeStateItem("econs"),
        AttributeStateItem("cons_err"),
        AttributeStateItem("ptens"),
        AttributeStateItem("vtens"),
        AttributeStateItem("press"),
        VolumeStateItem(),
        DomainStateItem(),
    ]
    log_name = "VERLET"

    def __init__(
        self,
        mmf,
        timestep=None,
        state=None,
        hooks=None,
        vel0=None,
        temp0=300 * kelvin,
        scalevel0=True,
        time0=None,
        ndof=None,
        counter0=None,
    ):
        # Assign initial arguments.
        self.ndof = ndof
        self.hooks = hooks

        if time0 is None:
            time0 = 0.0
        if counter0 is None:
            counter0 = 0

        self.pos = mmf.system.pos.copy()  # (N, 3) array
        self.rvecs = mmf.system.domain.rvecs.copy()
        self.masses = mmf.system.masses.copy()  # (N,) array
        self.timestep = timestep
        self.time = time0

        self._verify_hooks()

        # Set random initial velocities, with no center-of-mass velocity.
        if vel0 is None:
            self.vel = get_random_vel(temp0, scalevel0, self.masses)
            clean_momenta(self.pos, self.vel, self.masses, mmf.system.domain)
        else:
            self.vel = vel0.copy()

        # Initialize working arrays.
        self.gpos = np.zeros(self.pos.shape, float)
        self.delta = np.zeros(self.pos.shape, float)
        self.vtens = np.zeros((3, 3), float)

        # Initialize tracking of the error on the conserved quantity.
        self._cons_err_tracker = ConsErrTracker()

        # Initialize superclass.
        Iterative.__init__(self, mmf, state, self.hooks, counter0)

    def initialize(self):
        # Initialize Verlet algorithm.
        self.gpos[:] = 0.0
        self.delta[:] = 0.0
        self.mmf.update_pos(self.pos)
        self.epot = self.mmf.compute(self.gpos)
        self.acc = -self.gpos / self.masses.reshape(-1, 1)
        self.posold = self.pos.copy()

        # Allow for specialized initializations by the Verlet hooks.
        self.call_verlet_hooks("init")

        # Configure the number of degrees of freedom if needed.
        if self.ndof is None:
            self.ndof = np.size(self.pos)  # 3N degrees of freedom

        # Common post-processing of the initialization.
        self.compute_properties()
        Iterative.initialize(self)
        # Includes calls to conventional hooks.

    def propagate(self):
        self.call_verlet_hooks("pre")

        # Regular verlet step.
        self.acc = -self.gpos / self.masses.reshape(-1, 1)
        self.vel += 0.5 * self.acc * self.timestep
        self.pos += self.timestep * self.vel
        self.mmf.update_pos(self.pos)
        self.gpos[:] = 0.0
        self.vtens[:] = 0.0
        # Compute gradient and potential energy.
        self.epot = self.mmf.compute(self.gpos, self.vtens)
        self.acc = -self.gpos / self.masses.reshape(-1, 1)
        self.vel += 0.5 * self.acc * self.timestep
        self.ekin = self._compute_ekin()

        self.call_verlet_hooks("post")

        # Calculate the total position change.
        self.posnew = self.pos.copy()
        self.delta[:] = self.posnew - self.posold
        self.posold[:] = self.posnew

        # Common post-processing of a single step.
        self.time += self.timestep
        self.compute_properties()
        Iterative.propagate(self)

    def _compute_ekin(self):
        return np.sum(0.5 * (self.vel**2.0 * self.masses.reshape(-1, 1)))

    def compute_properties(self):
        self.rmsd_gpos = np.sqrt(np.mean(self.gpos**2))
        self.rmsd_delta = np.sqrt(np.mean(self.delta**2))
        self.ekin = self._compute_ekin()
        self.temp = (self.ekin / self.ndof) * (2.0 / boltzmann)
        self.etot = self.ekin + self.epot
        self.econs = self.etot

        for hook in self.hooks:
            if isinstance(hook, VerletHook):
                self.econs += hook.econs_correction

        self._cons_err_tracker.update(self.ekin, self.econs)
        self.cons_err = self._cons_err_tracker.get()

        if self.mmf.system.domain.nvec > 0:
            self.ptens = (
                np.dot(self.vel.T * self.masses, self.vel) - self.vtens
            ) / self.mmf.system.domain.volume
            self.press = np.trace(self.ptens) / 3.0

    def finalize(self):
        if log.do_medium:
            log.hline()

    def call_verlet_hooks(self, kind):
        # In this call, the state items are not updated.
        # The pre and post calls of the verlet hooks can rely on the specific implementation of the VerletIntegrator
        # and need not rely on the generic state item interface.
        with timer.section("%s special hooks" % self.log_name):
            for hook in self.hooks:
                if isinstance(hook, VerletHook) and hook.expects_call(self.counter):
                    if kind == "init":
                        hook.init(self)
                    elif kind == "pre":
                        hook.pre(self)
                    elif kind == "post":
                        hook.post(self)
                    else:
                        raise NotImplementedError

    def _add_default_hooks(self):
        if not any(isinstance(hook, VerletScreenLog) for hook in self.hooks):
            self.hooks.append(VerletScreenLog())

    def _verify_hooks(self):
        with log.section("ENSEM"):
            thermo = None
            index_thermo = 0
            baro = None
            index_baro = 0

            # Look for the presence of a thermostat and/or barostat.
            if hasattr(self.hooks, "__len__"):
                for index, hook in enumerate(self.hooks):
                    if hook.method == "thermostat":
                        thermo = hook
                        index_thermo = index
                    elif hook.method == "barostat":
                        baro = hook
                        index_baro = index
            elif self.hooks is not None:
                if self.hooks.method == "thermostat":
                    thermo = self.hooks
                elif self.hooks.method == "barostat":
                    baro = self.hooks

            # If both are present, delete them and generate TBCombination element.
            if thermo is not None and baro is not None:
                from .npt import TBCombination

                if log.do_warning:
                    log.warn(
                        "Both thermostat and barostat are present separately and will be merged"
                    )
                del self.hooks[max(index_thermo, index_thermo)]
                del self.hooks[min(index_thermo, index_baro)]
                self.hooks.append(TBCombination(thermo, baro))

            if hasattr(self.hooks, "__len__"):
                for hook in self.hooks:
                    if hook.name == "TBCombination":
                        thermo = hook.thermostat
                        baro = hook.barostat
            elif self.hooks is not None:
                if self.hooks.name == "TBCombination":
                    thermo = self.hooks.thermostat
                    baro = self.hooks.barostat

            if log.do_warning:
                if thermo is not None:
                    log(
                        "Temperature coupling achieved through "
                        + str(thermo.name)
                        + " thermostat"
                    )
                if baro is not None:
                    log(
                        "Pressure coupling achieved through "
                        + str(baro.name)
                        + " barostat"
                    )


class ConsErrTracker(object):
    """A class that tracks the errors on the conserved quantity.

    Given its superior numerical accuracy, the algorithm below is used to calculate the running average.
    Its properties are discussed in Donald Knuth"s Art of Computer Programming, vol. 2, p. 232, 3rd edition.
    """

    def __init__(self):
        self.counter = 0
        self.ekin_m = 0.0
        self.ekin_s = 0.0
        self.econs_m = 0.0
        self.econs_s = 0.0

    def update(self, ekin, econs):
        if self.counter == 0:
            self.ekin_m = ekin
            self.econs_m = econs
        else:
            ekin_tmp = ekin - self.ekin_m
            self.ekin_m += ekin_tmp / (self.counter + 1)
            self.ekin_s += ekin_tmp * (ekin - self.ekin_m)
            econs_tmp = econs - self.econs_m
            self.econs_m += econs_tmp / (self.counter + 1)
            self.econs_s += econs_tmp * (econs - self.econs_m)
        self.counter += 1

    def get(self):
        if self.counter > 1:
            # Returns the square root of the ratio of the variance
            # in kinetic energy to the variance in conserved energy.
            return np.sqrt(self.econs_s / self.ekin_s)
        return 0.0


class VerletHook(Hook):
    """Specialized Verlet hook.

    This is mainly used for the implementation of thermostats and barostats.
    """

    def __init__(self, start=0, step=1):
        """
        Parameters
        ----------
        start : int
            The first iteration at which this hook should be called.
        step : int
            The hook will be called every `step` iterations.

        """
        self.econs_correction = 0.0
        Hook.__init__(self, start=0, step=1)

    def __call__(self, iterative):
        pass

    def init(self, iterative):
        raise NotImplementedError

    def pre(self, iterative):
        raise NotImplementedError

    def post(self, iterative):
        raise NotImplementedError


class VerletScreenLog(Hook):
    """A screen logger for the Verlet algorithm."""

    def __init__(self, start=0, step=1):
        Hook.__init__(self, start, step)
        self.time0 = None

    def __call__(self, iterative):
        if log.do_medium:
            if self.time0 is None:
                self.time0 = time.time()
                if log.do_medium:
                    log.hline()
                    log(
                        "Cons.Err. =&the root of the ratio of the variance on the conserved quantity and the variance on the kinetic energy."
                    )
                    log("d-rmsd    =&the root-mean-square displacement of the nodes.")
                    log("g-rmsd    =&the root-mean-square gradient of the energy.")
                    log(
                        "counter  Cons.Err.       Temp     d-RMSD     g-RMSD   Walltime"
                    )
                    log.hline()
            log(
                "%7i %10.5f %s %s %s %10.1f"
                % (
                    iterative.counter,
                    iterative.cons_err,
                    log.temperature(iterative.temp),
                    log.length(iterative.rmsd_delta),
                    log.force(iterative.rmsd_gpos),
                    time.time() - self.time0,
                )
            )
