#!/usr/bin/env python

import numpy as np

from molmod import boltzmann, femtosecond

from micmec.log import log
from micmec.sampling.iterative import Iterative, StateItem
from micmec.sampling.utils import get_random_vel, clean_momenta, \
    get_ndof_internal_md, stabilized_cholesky_decomp
from micmec.sampling.verlet import VerletHook


__all__ = [
    "AndersenThermostat", 
    "BerendsenThermostat", 
    "LangevinThermostat",
    "CSVRThermostat", 
    "GLEThermostat", 
    "NHCThermostat", 
    "NHCAttributeStateItem",
]


class AndersenThermostat(VerletHook):
    name = "Andersen"
    kind = "stochastic"
    method = "thermostat"
    def __init__(self, temp, start=0, step=1, select=None, annealing=1.0):
        """
        This is an implementation of the Andersen thermostat. The method
        is described in:
            Andersen, H. C. J. Chem. Phys. 1980, 72, 2384-2393.

        **ARGUMENTS**
        temp
            The average temperature of the NVT ensemble
        
        **OPTIONAL ARGUMENTS**
        start
            The first iteration at which this hook is called
        step
            The number of iterations between two subsequent calls to this
            hook.
        select
            An array of atom indexes to indicate which atoms controlled by
            the thermostat.
        annealing
            After every call to this hook, the temperature is multiplied
            with this annealing factor. This effectively cools down the
            system.
        """
        self.temp = temp
        self.select = select
        self.annealing = annealing
        VerletHook.__init__(self, start, step)

    def init(self, iterative):
        # It is mandatory to zero the external momenta.
        clean_momenta(iterative.pos, iterative.vel, iterative.masses, iterative.mmf.system.cell)

    def pre(self, iterative, G1_add = None):
        # Andersen thermostat step before usual Verlet hook, since it largely affects the velocities.
        # Needed to correct the conserved quantity.
        ekin_before = iterative._compute_ekin()
        # Change the (selected) velocities.
        if self.select is None:
            iterative.vel[:] = get_random_vel(self.temp, False, iterative.masses)
        else:
            iterative.vel[self.select] = get_random_vel(self.temp, False, iterative.masses, self.select)
        # Zero any external momenta after choosing new velocities.
        clean_momenta(iterative.pos, iterative.vel, iterative.masses, iterative.mmf.system.cell)
        # Update the kinetic energy and the reference for the conserved quantity.
        ekin_after = iterative._compute_ekin()
        self.econs_correction += ekin_before - ekin_after
        # Optional annealing.
        self.temp *= self.annealing

    def post(self, iterative, G1_add = None):
        pass


class BerendsenThermostat(VerletHook):
    name = "Berendsen"
    kind = "deterministic"
    method = "thermostat"
    def __init__(self, temp, start=0, timecon=100*femtosecond, restart=False):
        """
        This is an implementation of the Berendsen thermostat. 
        The algorithm is described in:
            Berendsen, H. J. C.; Postma, J. P. M.; van Gunsteren, W. F.;
            Dinola, A.; Haak, J. R. J. Chem. Phys. 1984, 81, 3684-3690
        
        **ARGUMENTS**
        temp
            The temperature of thermostat.
        
        **OPTIONAL ARGUMENTS**
        start
            The step at which the thermostat becomes active.
        timecon
            The time constant of the Berendsen thermostat.
        restart
            Indicates whether the initalisation should be carried out.
        """
        self.temp = temp
        self.timecon = timecon
        self.restart = restart
        VerletHook.__init__(self, start, 1)

    def init(self, iterative):
        if not self.restart:
            # It is mandatory to zero the external momenta.
            clean_momenta(iterative.pos, iterative.vel, iterative.masses, iterative.mmf.system.cell)
        if iterative.ndof is None:
            iterative.ndof = get_ndof_internal_md(iterative.pos.shape[0], iterative.mmf.system.cell.nvec)

    def pre(self, iterative, G1_add = None):
        ekin = iterative.ekin
        temp_inst = 2.0*iterative.ekin/(boltzmann*iterative.ndof)
        c = np.sqrt(1+iterative.timestep/self.timecon*(self.temp/temp_inst-1))
        iterative.vel[:] = c*iterative.vel
        iterative.ekin = iterative._compute_ekin()
        self.econs_correction += (1-c**2)*ekin

    def post(self, iterative, G1_add = None):
        pass


class LangevinThermostat(VerletHook):
    name = "Langevin"
    kind = "stochastic"
    method = "thermostat"
    def __init__(self, temp, start=0, timecon=100*femtosecond):
        """
        This is an implementation of the Langevin thermostat. 
        The algorithm is described in:
            Bussi, G.; Parrinello, M. Phys. Rev. E 2007, 75, 056707
        
        **ARGUMENTS**
        temp
            The temperature of thermostat.
        
        **OPTIONAL ARGUMENTS**
        start
            The step at which the thermostat becomes active.
        timecon
            The time constant of the Langevin thermostat.
        """
        self.temp = temp
        self.timecon = timecon
        VerletHook.__init__(self, start, 1)

    def init(self, iterative):
        # It is mandatory to zero the external momenta.
        clean_momenta(iterative.pos, iterative.vel, iterative.masses, iterative.mmf.system.cell)

    def pre(self, iterative, G1_add = None):
        ekin0 = iterative.ekin
        # Actual update.
        self.thermo(iterative)
        ekin1 = iterative.ekin
        self.econs_correction += ekin0-ekin1

    def post(self, iterative, G1_add = None):
        ekin0 = iterative.ekin
        # Actual update.
        self.thermo(iterative)
        ekin1 = iterative.ekin
        self.econs_correction += ekin0-ekin1

    def thermo(self, iterative):
        c1 = np.exp(-iterative.timestep/self.timecon/2)
        c2 = np.sqrt((1.0-c1**2)*self.temp*boltzmann/iterative.masses).reshape(-1,1)
        iterative.vel[:] = c1*iterative.vel + c2*np.random.normal(0, 1, iterative.vel.shape)
        iterative.ekin = iterative._compute_ekin()


class CSVRThermostat(VerletHook):
    name = "CSVR"
    kind = "stochastic"
    method = "thermostat"
    def __init__(self, temp, start=0, timecon=100*femtosecond):
        """
        This is an implementation of the CSVR thermostat. The equations are
        derived in:
            Bussi, G.; Donadio, D.; Parrinello, M. J. Chem. Phys. 2007,
            126, 014101
        and the implementation (used here) is derived in:
            Bussi, G.; Parrinello, M. Comput. Phys. Commun. 2008, 179, 26-29.

        **ARGUMENTS**
        temp
            The temperature of thermostat.

        **OPTIONAL ARGUMENTS**
        start
            The step at which the thermostat becomes active.
        timecon
            The time constant of the CSVR thermostat.
        """
        self.temp = temp
        self.timecon = timecon
        VerletHook.__init__(self, start, 1)

    def init(self, iterative):
        # It is mandatory to zero the external momenta.
        clean_momenta(iterative.pos, iterative.vel, iterative.masses, iterative.mmf.system.cell)
        if iterative.ndof is None:
            iterative.ndof = get_ndof_internal_md(iterative.pos.shape[0], iterative.mmf.system.cell.nvec)
        self.kin = 0.5*iterative.ndof*boltzmann*self.temp

    def pre(self, iterative, G1_add = None):
        c = np.exp(-iterative.timestep/self.timecon)
        R = np.random.normal(0, 1)
        S = (np.random.normal(0, 1, iterative.ndof-1)**2).sum()
        iterative.ekin = iterative._compute_ekin()
        fact = (1-c)*self.kin/iterative.ndof/iterative.ekin
        alpha = np.sign(R+np.sqrt(c/fact))*np.sqrt(c + (S+R**2)*fact + 2*R*np.sqrt(c*fact))
        iterative.vel[:] = alpha*iterative.vel
        iterative.ekin_new = alpha**2*iterative.ekin
        self.econs_correction += (1-alpha**2)*iterative.ekin
        iterative.ekin = iterative.ekin_new

    def post(self, iterative, G1_add = None):
        pass


class GLEThermostat(VerletHook):
    name = "GLE"
    kind = "stochastic"
    method = "thermostat"
    def __init__(self, temp, a_p, c_p=None, start=0):
        """
        This hook implements the coloured noise thermostat. The equations
        are derived in:
            Ceriotti, M.; Bussi, G.; Parrinello, M J. Chem. Theory Comput.
            2010, 6, 1170-1180.
        
        **ARGUMENTS**
        temp
            The temperature of thermostat.
        a_p
            Square drift matrix, with elements fitted to the specific problem.
        
        **OPTIONAL ARGUMENTS**
        c_p
            Square static covariance matrix. In equilibrium, its elements are fixed.
            For non-equilibrium dynamics, its elements should be fitted.
        start
            The step at which the thermostat becomes active.
        """
        self.temp = temp
        self.ns = int(a_p.shape[0]-1)
        self.a_p = a_p
        self.c_p = c_p
        if self.c_p is None:
            # Assume equilibrium dynamics if c_p is not provided.
            self.c_p = boltzmann*self.temp*np.eye(self.ns+1)
        VerletHook.__init__(self, start, 1)

    def init(self, iterative):
        # It is mandatory to zero the external momenta.
        clean_momenta(iterative.pos, iterative.vel, iterative.masses, iterative.mmf.system.cell)
        # Initialize the additional momenta.
        self.s = 0.5*boltzmann*self.temp*np.random.normal(size=(self.ns, iterative.pos.size))
        # Determine the update matrices.
        eigval, eigvec = np.linalg.eig(-self.a_p * iterative.timestep/2)
        self.t = np.dot(eigvec*np.exp(eigval), np.linalg.inv(eigvec)).real
        self.S = stabilized_cholesky_decomp(self.c_p - np.dot(np.dot(self.t,self.c_p),self.t.T)).real
        # Store the number of atoms for later use.
        self.n_atoms = iterative.pos.shape[0]

    def pre(self, iterative, G1_add = None):
        self.thermo(iterative)

    def post(self, iterative, G1_add = None):
        self.thermo(iterative)

    def thermo(self, iterative):
        ekin0 = iterative.ekin
        # Define a (3N,) vector of rescaled momenta.
        p = np.dot(np.diag(np.sqrt(iterative.masses)),iterative.vel).reshape(-1)
        # Extend the s to include the real momenta.
        s_extended_old = np.vstack([p, self.s])
        # Update equation.
        s_extended_new = np.dot(self.t, s_extended_old) + np.dot(self.S, np.random.normal(size = (self.ns+1, 3*self.n_atoms)))
        # Store the new variables in the correct place.
        iterative.vel[:] = np.dot(np.diag(np.sqrt(1.0/iterative.masses)),s_extended_new[0,:].reshape((self.n_atoms,3)))
        self.s[:] = s_extended_new[1:s_extended_new.shape[0],:]
        # Update the kinetic energy.
        iterative.ekin = iterative._compute_ekin()
        # Update the conserved quantity.
        ekin1 = iterative.ekin
        self.econs_correction += ekin0-ekin1


class NHChain(object):
    def __init__(self, length, timestep, temp, ndof, pos0, vel0, timecon=100*femtosecond):
        # Parameters.
        self.length = length
        self.timestep = timestep
        self.temp = temp
        self.timecon = timecon
        # Verify whether positions and velocities are taken from a restart.
        self.restart_pos = False
        self.restart_vel = False
        if pos0 is not None: self.restart_pos = True
        if vel0 is not None: self.restart_vel = True

        if ndof>0: # Avoid setting self.masses with zero gaussian-width in set_ndof if ndof=0.
            self.set_ndof(ndof)

        # Allocate degrees of freedom.
        if self.restart_pos: self.pos = pos0.copy()
        else: self.pos = np.zeros(length)
        if self.restart_vel: self.vel = vel0.copy()
        else: self.vel = np.zeros(length)

    def set_ndof(self, ndof):
        # Set the masses according to the time constant.
        self.ndof = ndof
        afreq = 2*np.pi/self.timecon
        self.masses = np.ones(self.length)*(boltzmann*self.temp/afreq**2)
        self.masses[0] *= ndof
        if not self.restart_vel: self.vel = self.get_random_vel_therm()

    def get_random_vel_therm(self):
        # Generate random velocities for the thermostat velocities using a Gaussian distribution.
        shape = self.length
        return np.random.normal(0, np.sqrt(self.masses*boltzmann*self.temp), shape)/self.masses

    def __call__(self, ekin, vel, G1_add):
        def do_bead(k, ekin):
            # Compute g.
            if k == 0:
                # Coupling with atoms (and barostat).
                # L = ndof (+d(d+1)/2 (aniso NPT) / +1 (iso NPT)) because of equidistant time steps.
                g = 2*ekin - self.ndof*self.temp*boltzmann
                if G1_add is not None:
                    # add pressure contribution to g1
                    g += G1_add
            else:
                # Coupling between beads.
                g = self.masses[k-1]*self.vel[k-1]**2 - self.temp*boltzmann
            g /= self.masses[k]

            # Liouville operators on relevant part of the chain.
            if k == self.length-1:
                # iL G_k h/4
                self.vel[k] += g*self.timestep/4
            else:
                # iL vxi_{k-1} h/8
                self.vel[k] *= np.exp(-self.vel[k+1]*self.timestep/8)
                # iL G_k h/4
                self.vel[k] += g*self.timestep/4
                # iL vxi_{k-1} h/8
                self.vel[k] *= np.exp(-self.vel[k+1]*self.timestep/8)

        # Loop over chain in reverse order.
        for k in range(self.length-1, -1, -1):
            do_bead(k, ekin)

        # iL xi (all) h/2
        self.pos += self.vel*self.timestep/2
        # iL Cv (all) h/2
        factor = np.exp(-self.vel[0]*self.timestep/2)
        vel *= factor
        ekin *= factor**2

        # Loop over chain in forward order.
        for k in range(0, self.length):
            do_bead(k, ekin)
        return vel, ekin

    def get_econs_correction(self):
        kt = boltzmann*self.temp
        # Correction due to the thermostat.
        return 0.5*(self.vel**2*self.masses).sum() + kt*(self.ndof*self.pos[0] + self.pos[1:].sum())


class NHCThermostat(VerletHook):
    name = "NHC"
    kind = "deterministic"
    method = "thermostat"
    def __init__(self, temp, start=0, timecon=100*femtosecond, chainlength=3, chain_pos0=None, chain_vel0=None, restart=False):
        """
        This hook implements the Nose-Hoover chain thermostat. The equations
        are derived in:
            Martyna, G. J.; Klein, M. L.; Tuckerman, M. J. Chem. Phys. 1992,
            97, 2635-2643.
        The implementation (used here) of a symplectic integrator of the
        Nose-Hoover chain thermostat is discussed in:
            Martyna, G. J.;  Tuckerman, M. E.;  Tobias, D. J.;  Klein,
            M. L. Mol. Phys. 1996, 87, 1117-1157.

        **ARGUMENTS**
        temp
            The temperature of thermostat.

        **OPTIONAL ARGUMENTS**
        start
            The step at which the thermostat becomes active.
        timecon
            The time constant of the Nose-Hoover thermostat.
        chainlength
            The number of beads in the Nose-Hoover chain.
        chain_pos0
            The initial thermostat chain positions.
        chain_vel0
            The initial thermostat chain velocities.
        restart
            Indicates whether the initalisation should be carried out.
        """
        self.temp = temp
        self.restart = restart
        # At this point, the timestep and the number of degrees of freedom are not known yet.
        self.chain = NHChain(chainlength, 0.0, temp, 0, chain_pos0, chain_vel0, timecon)
        VerletHook.__init__(self, start, 1)

    def init(self, iterative):
        if not self.restart:
            # It is mandatory to zero the external momenta.
            clean_momenta(iterative.pos, iterative.vel, iterative.masses, iterative.mmf.system.cell)
        # If needed, determine the number of _internal_ degrees of freedom.
        if iterative.ndof is None:
            iterative.ndof = get_ndof_internal_md(iterative.pos.shape[0], iterative.mmf.system.cell.nvec)
        # Configure the chain.
        self.chain.timestep = iterative.timestep
        self.chain.set_ndof(iterative.ndof)

    def pre(self, iterative, G1_add = None):
        velnew, iterative.ekin = self.chain(iterative.ekin, iterative.vel, G1_add)
        iterative.vel[:] = velnew

    def post(self, iterative, G1_add = None):
        velnew, iterative.ekin = self.chain(iterative.ekin, iterative.vel, G1_add)
        iterative.vel[:] = velnew
        self.econs_correction = self.chain.get_econs_correction()


class NHCAttributeStateItem(StateItem):
    def __init__(self, attr):
        StateItem.__init__(self, "thermo_" + attr)
        self.attr = attr

    def get_value(self, iterative):
        chain = None
        from yaff.sampling.npt import TBCombination
        for hook in iterative.hooks:
            if isinstance(hook, NHCThermostat):
                chain = hook.chain
                break
            elif isinstance(hook, TBCombination):
                if isinstance(hook.thermostat, NHCThermostat):
                    chain = hook.thermostat.chain
                break
        if chain is None:
            raise TypeError("Iterative does not contain a NHCThermostat hook.")
        return getattr(chain, self.attr)

    def copy(self):
        return self.__class__(self.attr)


