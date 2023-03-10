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


"""Geometry optimization."""

import time
import numpy as np

from numpy.linalg import eigh
from molmod.minimizer import ConjugateGradient, NewtonLineSearch, Minimizer

from micmec.log import log
from micmec.sampling.iterative import (
    Iterative,
    AttributeStateItem,
    PosStateItem,
    VolumeStateItem,
    DomainStateItem,
    EPotContribStateItem,
    Hook,
)


__all__ = [
    "OptScreenLog",
    "BaseOptimizer",
    "CGOptimizer",
    "BFGSHessianModel",
    "SR1HessianModel",
    "QNOptimizer",
    "solve_trust_radius",
]


class OptScreenLog(Hook):
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
                        "Conv.val. =&the highest ratio of a convergence criterion over its threshold."
                    )
                    log(
                        "N         =&the number of convergence criteria that is not met."
                    )
                    log(
                        "Worst     =&the name of the convergence criterion that is worst."
                    )
                    log("counter  Conv.val.  N           Worst     Energy   Walltime")
                    log.hline()
            log(
                "%7i % 10.3e %2i %15s %s %10.1f"
                % (
                    iterative.counter,
                    iterative.dof.conv_val,
                    iterative.dof.conv_count,
                    iterative.dof.conv_worst,
                    log.energy(iterative.epot),
                    time.time() - self.time0,
                )
            )


class BaseOptimizer(Iterative):
    default_state = [
        AttributeStateItem("counter"),
        AttributeStateItem("epot"),
        PosStateItem(),
        VolumeStateItem(),
        DomainStateItem(),
        EPotContribStateItem(),
    ]
    log_name = "XXOPT"

    def __init__(self, dof, state=None, hooks=None, counter0=0):
        """
        Parameters
        ----------
        dof : micmec.sampling.dof.DOF object
            A specification of the degrees of freedom.
            The convergence criteria are also part of this argument.
        state : list
            A list with state items.
            State items are simple objects that take or derive a property from the current state of the iterative
            algorithm.
        hooks :
            A function (or a list of functions) that is called after every iterative.
        counter0 :
            The counter value associated with the initial state.

        """
        self.dof = dof
        Iterative.__init__(self, dof.mmf, state, hooks, counter0)

    def _add_default_hooks(self):
        if not any(isinstance(hook, OptScreenLog) for hook in self.hooks):
            self.hooks.append(OptScreenLog())

    def fun(self, x, do_gradient=False):
        if do_gradient:
            self.epot, gx = self.dof.fun(x, True)
            return self.epot, gx
        else:
            self.epot = self.dof.fun(x, False)
            return self.epot

    def initialize(self):
        # The first call to check_convergence will never flag convergence, but
        # it is need to keep track of some convergence criteria.
        self.dof.check_convergence()
        Iterative.initialize(self)

    def propagate(self):
        self.dof.check_convergence()
        Iterative.propagate(self)
        return self.dof.converged

    def finalize(self):
        if log.do_medium:
            self.dof.log()
            log.hline()


class CGOptimizer(BaseOptimizer):
    """A conjugate gradient optimizer."""

    log_name = "CGOPT"

    def __init__(self, dof, state=None, hooks=None, counter0=0):
        self.minimizer = Minimizer(
            dof.x0,
            self.fun,
            ConjugateGradient(),
            NewtonLineSearch(),
            None,
            None,
            anagrad=True,
            verbose=False,
        )
        BaseOptimizer.__init__(self, dof, state, hooks, counter0)

    def initialize(self):
        self.minimizer.initialize()
        BaseOptimizer.initialize(self)

    def propagate(self):
        success = self.minimizer.propagate()
        self.x = self.minimizer.x
        if not success:
            if log.do_warning:
                log.warn(
                    "Line search failed in optimizer. Aborting optimization. \
                            This is probably due to a dicontinuity in the energy or the forces. \
                            Check the truncation of the non-bonding interactions and the Ewald summation parameters."
                )
            return True
        return BaseOptimizer.propagate(self)


class HessianModel(object):
    def __init__(self, ndof, hessian0=None):
        """
        Parameters
        ----------
        ndof : int
            The number of degrees of freedom.
        hessian0 : optional
            An initial guess for the hessian.

        """
        self.ndof = ndof
        if hessian0 is None:
            self.hessian = np.identity(ndof, float)
        else:
            self.hessian = hessian0.copy()
            if self.hessian.shape != (ndof, ndof):
                raise TypeError(
                    "Incorrect shape of the initial hessian in quasi-newton method."
                )

    def get_spectrum(self):
        return eigh(self.hessian)


class BFGSHessianModel(HessianModel):
    def update(self, dx, dg):
        tmp = np.dot(self.hessian, dx)
        hmax = abs(self.hessian).max()
        # Only compute updates if the denominators do not blow up
        denom1 = np.dot(dx, tmp)
        if hmax * denom1 <= 1e-5 * abs(tmp).max() ** 2:
            if log.do_high:
                log(
                    "Skipping BFGS update because denom1=%10.3e is not positive enough."
                    % denom1
                )
            return False
        denom2 = np.dot(dg, dx)
        if hmax * denom2 <= 1e-5 * abs(dg).max() ** 2:
            if log.do_high:
                log(
                    "Skipping BFGS update because denom2=%10.3e is not positive enough."
                    % denom2
                )
            return False
        if log.do_debug:
            log(
                "Updating BFGS Hessian.    denom1=%10.3e   denom2=%10.3e"
                % (denom1, denom2)
            )
        self.hessian -= np.outer(tmp, tmp) / denom1
        self.hessian += np.outer(dg, dg) / denom2
        return True


class SR1HessianModel(HessianModel):
    def update(self, dx, dg):
        tmp = dg - np.dot(self.hessian, dx)

        denom = np.dot(tmp, dx)
        if abs(denom) > 1e-5 * np.linalg.norm(dx) * np.linalg.norm(tmp):
            if log.do_debug:
                log("Updating SR1 Hessian.       denom=%10.3e" % denom)
            self.hessian += np.outer(tmp, tmp) / denom
            return True
        else:
            if log.do_high:
                log(
                    "Skipping SR1 update because denom=%10.3e is not big enough."
                    % denom
                )
            return False


class QNOptimizer(BaseOptimizer):
    """A Quasi-Newtonian optimizer."""

    log_name = "QNOPT"

    def __init__(
        self,
        dof,
        state=None,
        hooks=None,
        counter0=0,
        trust_radius=1.0,
        small_radius=1e-5,
        too_small_radius=1e-10,
        hessian0=None,
    ):
        """
        Parameters
        ----------
        dof : micmec.sampling.dof.DOF object
            A specification of the degrees of freedom.
            The convergence criteria are also part of this argument.
        state : list
            A list with state items.
            State items are simple objects that take or derive a property from the current state of the iterative
            algorithm.
        hooks :
            A function (or a list of functions) that is called after every iterative.
        counter0 :
            The counter value associated with the initial state.
        trust_radius : float
            The initial value for the trust radius.
            It is adapted by the algorithm after every step.
            The adapted trust radius is never allowed to increase above this initial value.
        small_radius : float
            If the trust radius goes below this limit, the decrease in energy is no longer essential.
            Instead a decrease in the norm of the gradient is used to accept/reject a step.
        too_small_radius : float
            If the trust radius becomes smaller than this parameter, the optimizer gives up.
            Insanely small trust radii are typical for potential energy surfaces that are not entirely smooth.
        hessian0 :
            An initial guess for the Hessian.

        """
        self.x_old = dof.x0
        self.hessian = SR1HessianModel(len(dof.x0), hessian0)
        self.trust_radius = trust_radius
        self.initial_trust_radius = trust_radius
        self.small_radius = small_radius
        self.too_small_radius = too_small_radius
        BaseOptimizer.__init__(self, dof, state, hooks, counter0)

    def initialize(self):
        self.f_old, self.g_old = self.fun(self.dof.x0, True)
        self.x, self.f, self.g = self.make_step()
        BaseOptimizer.initialize(self)

    def propagate(self):
        # Update the Hessian.
        assert self.g is not self.g_old
        assert self.x is not self.x_old
        hessian_safe = self.hessian.update(self.x - self.x_old, self.g - self.g_old)
        if not hessian_safe:
            # Reset the Hessian completely.
            if log.do_high:
                log("Resetting hessian due to failed update.")
            self.hessian = SR1HessianModel(len(self.x))
            self.trust_radius = self.initial_trust_radius
        # Move new to old.
        self.x_old = self.x
        self.f_old = self.f
        self.g_old = self.g
        # Compute a step.
        self.x, self.f, self.g = self.make_step()
        return BaseOptimizer.propagate(self)

    def make_step(self):
        # Get relevant hessian information.
        evals, evecs = self.hessian.get_spectrum()
        if log.do_high:
            log(" lowest eigen value: %7.1e" % evals.min())
            log("highest eigen value: %7.1e" % evals.max())
        # Convert gradient to eigenbasis.
        grad_eigen = np.dot(evecs.T, self.g_old)

        while True:
            # Find the step with the given radius. If the hessian is positive definite and the unconstrained step is
            # smaller than the trust radius, this step is returned.
            delta_eigen = solve_trust_radius(grad_eigen, evals, self.trust_radius)
            radius = np.linalg.norm(delta_eigen)

            # Convert the step to user basis.
            delta_x = np.dot(evecs, delta_eigen)

            # Compute the function and gradient at the new position.
            x = self.x_old + delta_x
            f, g = self.fun(x, True)

            # Compute the change in function value.
            delta_f = f - self.f_old
            # Compute the change in norm of the gradient.
            delta_norm_g = np.linalg.norm(g) - np.linalg.norm(self.g_old)
            # must_shrink is a parameter to control the trust radius.
            must_shrink = False

            if delta_f > 0:
                # The function must decrease, if not the trust radius is too big.
                if log.do_high:
                    log("Function increases.")
                must_shrink = True

            if self.trust_radius < self.small_radius and delta_norm_g > 0:
                # When the trust radius becomes small, the numerical noise on the energy may be too large to detect
                # an increase energy.
                # In that case the norm of the gradient is used instead.
                if log.do_high:
                    log("Gradient norm increases.")
                must_shrink = True

            if must_shrink:
                self.trust_radius *= 0.5
                while self.trust_radius >= radius:
                    self.trust_radius *= 0.5
                if self.trust_radius < self.too_small_radius:
                    raise RuntimeError(
                        "The trust radius becomes too small. Is the potential energy surface smooth?"
                    )
            else:
                # If we get here, we are done with the trust radius loop.
                if log.do_high:
                    log.hline()
                # It is fine to increase the trust radius a little after a successful step.
                if self.trust_radius < self.initial_trust_radius:
                    self.trust_radius *= 2.0
                # Return the results of the successful step.
                return x, f, g


def solve_trust_radius(grad, evals, radius, threshold=1e-5):
    """Find a step in eigenspace with the given radius."""
    # First try an unconstrained step if the eigenvalues are all strictly positive.
    if evals.min() > 0:
        step = -grad / evals
        if np.linalg.norm(step) <= radius:
            return step

    # Define some auxiliary functions.
    def compute_step(ridge):
        return -grad / (evals + ridge)

    def compute_error(ridge):
        return np.linalg.norm(compute_step(ridge)) - radius

    # The ultimate lower bound that we'd rather avoid.
    ridge_min = -evals.min()

    def find_edge(alpha, sign):
        while True:
            ridge = ridge_min + alpha
            error = compute_error(ridge)
            if sign * error < 0:
                return ridge, error
            if sign > 0:
                alpha *= 2
            else:
                alpha /= 2

    # Find a proper lower bound, error > 0.
    a = find_edge(min(1e1, abs(evals.max())), -1)
    # Find an upper bound, error < 0.
    b = find_edge(max(1e-5, abs(ridge_min)), 1)

    # Bisection algorithm to find root.
    error = np.inf
    while abs(error) > radius * threshold:
        ridge = (a[1] * a[0] - b[1] * b[0]) / (a[1] - b[1])
        error = compute_error(ridge)
        c = ridge, error
        if error > 0 and a[1] > 0:
            a = c
        else:
            b = c

    # Best guess.
    ridge = c[0]
    return compute_step(ridge)
