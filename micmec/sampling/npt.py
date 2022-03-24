#!/usr/bin/env python

import numpy as np

from molmod import boltzmann, femtosecond, kjmol, bar, atm

from ..log import log, timer
from .utils import get_random_vel, domain_symmetrize, get_random_vel_press, \
    get_ndof_internal_md, clean_momenta, get_ndof_baro
from .verlet import VerletHook
from .iterative import StateItem


__all__ = [
    "TBCombination", "McDonaldBarostat", "BerendsenBarostat", "LangevinBarostat",
    "MTKBarostat", "PRBarostat", "TadmorBarostat", "MTKAttributeStateItem"
]

class TBCombination(VerletHook):
    name = "TBCombination"
    def __init__(self, thermostat, barostat, start=0):
        """
        VerletHook combining an arbitrary Thermostat and Barostat instance, which
        ensures these instances are called in the correct succession, and possible
        coupling between both is handled correctly.

        **ARGUMENTS**
        thermostat
            A Thermostat instance.
        barostat
            A Barostat instance.
        """
        self.thermostat = thermostat
        self.barostat = barostat
        self.start = start
        # Verify if thermostat and barostat instances are currently supported in Yaff/MicMec.
        if not self.verify():
            self.barostat = thermostat
            self.thermostat = barostat
            if not self.verify():
                raise TypeError("The Thermostat or Barostat instance is not supported (yet)")
        self.step_thermo = self.thermostat.step
        self.step_baro = self.barostat.step
        VerletHook.__init__(self, start, min(self.step_thermo, self.step_baro))

    def init(self, iterative):
        # Verify whether ndof is given as an argument.
        set_ndof = iterative.ndof is not None
        # Initialize the thermostat and barostat separately.
        self.thermostat.init(iterative)
        self.barostat.init(iterative)
        # Ensure ndof = 3N if the center of mass movement is not suppressed and ndof is not determined by the user.
        from .nvt import LangevinThermostat, GLEThermostat
        p_cm_fluct = isinstance(self.thermostat, LangevinThermostat) or isinstance(self.thermostat, GLEThermostat) or isinstance(self.barostat, LangevinBarostat)
        if (not set_ndof) and p_cm_fluct:
            iterative.ndof = iterative.pos.size
        # Variables which will determine the coupling between thermostat and barostat.
        self.chainvel0 = None
        self.G1_add = None

    def pre(self, iterative):
        # Determine whether the barostat should be called.
        if self.expectscall(iterative, "baro"):
            from .nvt import NHCThermostat
            if isinstance(self.thermostat, NHCThermostat):
                # In case the barostat is coupled with a NHC thermostat:
                # v_{xi,1} is needed to update v_g.
                self.chainvel0 = self.thermostat.chain.vel[0]
            # Actual barostat update.
            self.barostat.pre(iterative, self.chainvel0)
        # Determine whether the thermostat should be called.
        if self.expectscall(iterative, "thermo"):
            if isinstance(self.barostat, MTKBarostat):
                # In case the thermostat is coupled with a MTK barostat:
                # update equation of v_{xi,1} is altered via G_1.
                self.G1_add = self.barostat.add_press_cont()
            # Actual thermostat update.
            self.thermostat.pre(iterative, self.G1_add)

    def post(self, iterative):
        # Determine whether the thermostat should be called.
        if self.expectscall(iterative, "thermo"):
            if isinstance(self.barostat, MTKBarostat):
                # In case the thermostat is coupled with a MTK barostat:
                # update equation of v_{xi,1} is altered via G_1.
                self.G1_add = self.barostat.add_press_cont()
            # Actual thermostat update.
            self.thermostat.post(iterative, self.G1_add)
        # Determine whether the barostat should be called.
        if self.expectscall(iterative, "baro"):
            from .nvt import NHCThermostat, LangevinThermostat
            if isinstance(self.thermostat, NHCThermostat):
                # In case the barostat is coupled with a NHC thermostat:
                # v_{xi,1} is needed to update v_g.
                self.chainvel0 = self.thermostat.chain.vel[0]
            # Actual barostat update.
            self.barostat.post(iterative, self.chainvel0)
        # Update the correction on E_cons due to thermostat and barostat.
        self.econs_correction = self.thermostat.econs_correction + self.barostat.econs_correction
        if isinstance(self.thermostat, NHCThermostat):
            if isinstance(self.barostat, MTKBarostat) and self.barostat.baro_thermo is not None: pass
            else:
                # Extra correction necessary if particle NHC thermostat is used to thermostat the barostat.
                kt = boltzmann*self.thermostat.temp
                baro_ndof = self.barostat.baro_ndof
                self.econs_correction += baro_ndof*kt*self.thermostat.chain.pos[0]

    def expectscall(self, iterative, kind):
        # Returns whether the thermostat/barostat should be called in this iteration.
        if kind == "thermo":
            return iterative.counter >= self.start and (iterative.counter - self.start) % self.step_thermo == 0
        if kind == "baro":
            return iterative.counter >= self.start and (iterative.counter - self.start) % self.step_baro == 0

    def verify(self):
        # Returns whether the thermostat and barostat instances are currently supported by Yaff/MicMec.
        from .nvt import AndersenThermostat, NHCThermostat, LangevinThermostat, BerendsenThermostat, CSVRThermostat, GLEThermostat
        thermo_correct = False
        baro_correct = False
        thermo_list = [AndersenThermostat, NHCThermostat, LangevinThermostat, BerendsenThermostat, CSVRThermostat, GLEThermostat]
        baro_list = [McDonaldBarostat, BerendsenBarostat, LangevinBarostat, MTKBarostat, PRBarostat, TadmorBarostat]
        if any(isinstance(self.thermostat, thermo) for thermo in thermo_list):
            thermo_correct = True
        if any(isinstance(self.barostat, baro) for baro in baro_list):
            baro_correct = True
        return (thermo_correct and baro_correct)


class McDonaldBarostat(VerletHook):
    name = "McDonald"
    kind = "stochastic"
    method = "barostat"
    def __init__(self, temp, press, start=0, step=1, amp=1e-3):
        """
           Warning: this code is not fully tested yet!
           **ARGUMENTS**
           temp
                The average temperature of the NpT ensemble
           press
                The external pressure of the NpT ensemble
           **OPTIONAL ARGUMENTS**
           start
                The first iteration at which this hook is called
           step
                The number of iterations between two subsequent calls to this
                hook.
           amp
                The amplitude of the changes in the logarithm of the volume.
        """
        self.temp = temp
        self.press = press
        self.amp = amp
        self.dim = 3
        self.baro_ndof = 1
        VerletHook.__init__(self, start, step)

    def init(self, iterative):
        pass

    def pre(self, iterative, chainvel0 = None):
        pass

    def post(self, iterative, chainvel0 = None):
        def compute(pos, rvecs):
            iterative.pos[:] = pos
            iterative.gpos[:] = 0.0
            iterative.mmf.update_rvecs(rvecs)
            iterative.mmf.update_pos(pos)
            iterative.epot = iterative.mmf.compute(iterative.gpos)
            iterative.acc = -iterative.gpos/iterative.masses.reshape(-1,1)

        nnodes = iterative.mmf.system.nnodes
        # Change the logarithm of the volume isotropically.
        scale = np.exp(np.random.uniform(-self.amp, self.amp))
        # Keep track of old state.
        vol0 = iterative.mmf.system.domain.volume
        epot0 = iterative.epot
        rvecs0 = iterative.mmf.system.domain.rvecs.copy()
        pos0 = iterative.pos.copy()
        # Scale the system and recompute the energy.
        compute(pos0*scale, rvecs0*scale)
        epot1 = iterative.epot
        vol1 = iterative.mmf.system.domain.volume
        # Compute the acceptance ratio.
        beta = 1/(boltzmann*self.temp)
        arg = (epot1 - epot0 + self.press*(vol1 - vol0) - (nnodes+1)/beta*np.log(vol1/vol0))
        accepted = arg < 0 or np.random.uniform(0, 1) < np.exp(-beta*arg)
        if accepted:
            # Add a correction to the conserved quantity.
            self.econs_correction += epot0 - epot1
        else:
            # Revert the domain and the positions in the original state.
            compute(pos0, rvecs0)


class BerendsenBarostat(VerletHook):
    name = "Berendsen"
    kind = "deterministic"
    method = "barostat"
    def __init__(self, mmf, temp, press, start=0, step=1, timecon=1000*femtosecond, beta=4.57e-5/bar, anisotropic=True, vol_constraint=False, restart=False):
        """
        This hook implements the Berendsen barostat. The equations are derived in:
            Berendsen, H. J. C.; Postma, J. P. M.; van Gunsteren, W. F.;
            Dinola, A.; Haak, J. R. J. Chem. Phys. 1984, 81, 3684-3690

        **ARGUMENTS**
        mmf
            A MicMecForceField instance.
        temp
            The temperature of thermostat.
        press
            The applied pressure for the barostat.

        **OPTIONAL ARGUMENTS**
        start
            The step at which the thermostat becomes active.
        timecon
            The time constant of the Berendsen barostat.
        beta
            The isothermal compressibility, conventionally the compressibility of liquid water.
        anisotropic
            Defines whether anisotropic domain fluctuations are allowed.
        vol_constraint
            Defines whether the volume is allowed to fluctuate.
        """
        self.temp = temp
        self.press = press
        self.timecon_press = timecon
        self.beta = beta
        self.mass_press = 3.0*timecon/beta
        self.anisotropic = anisotropic
        self.vol_constraint = vol_constraint
        self.dim = mmf.system.domain.nvec
        # Determine the number of degrees of freedom associated with the unit domain.
        self.baro_ndof = get_ndof_baro(self.dim, self.anisotropic, self.vol_constraint)
        if self.anisotropic:
            # Symmetrize the domain tensor.
            domain_symmetrize(mmf)
        self.domain = mmf.system.domain.rvecs.copy()
        self.restart = restart
        VerletHook.__init__(self, start, step)

    def init(self, iterative):
        self.timestep_press = iterative.timestep
        if not self.restart:
            clean_momenta(iterative.pos, iterative.vel, iterative.masses, iterative.mmf.system.domain)
        # Compute gpos and vtens, since they differ
        # after symmetrising the domain tensor.
        iterative.gpos[:] = 0.0
        iterative.vtens[:] = 0.0
        iterative.epot = iterative.mmf.compute(iterative.gpos,iterative.vtens)
        if iterative.ndof is None:
            iterative.ndof = get_ndof_internal_md(iterative.pos.shape[0], iterative.mmf.system.domain.nvec)
        # Rescaling of the barostat mass, to be in accordance with Langevin and MTTK.
        self.mass_press *= np.sqrt(iterative.ndof)

    def pre(self, iterative, chainvel0 = None):
        pass

    def post(self, iterative, chainvel0 = None):
        # For bookkeeping purposes.
        epot0 = iterative.epot
        # Calculation of the internal pressure tensor.
        ptens = (np.dot(iterative.vel.T*iterative.masses, iterative.vel) - iterative.vtens)/iterative.mmf.system.domain.volume
        # Determination of mu.
        dmu = self.timestep_press/self.mass_press*(self.press*np.eye(3)-ptens)
        if self.vol_constraint:
            dmu -= np.trace(dmu)/self.dim*np.eye(self.dim)
        mu = np.eye(3) - dmu
        mu = 0.5*(mu+mu.T)
        if not self.anisotropic:
            mu = ((np.trace(mu)/3.0)**(1.0/3.0))*np.eye(3)
        # Updating the positions and domain vectors.
        pos_new = np.dot(iterative.pos, mu)
        rvecs_new = np.dot(iterative.rvecs, mu)
        iterative.mmf.update_pos(pos_new)
        iterative.pos[:] = pos_new
        iterative.mmf.update_rvecs(rvecs_new)
        iterative.rvecs[:] = rvecs_new
        # Calculation of the virial tensor.
        iterative.gpos[:] = 0.0
        iterative.vtens[:] = 0.0
        iterative.epot = iterative.mmf.compute(iterative.gpos, iterative.vtens)
        epot1 = iterative.epot
        self.econs_correction += epot0 - epot1


class LangevinBarostat(VerletHook):
    name = "Langevin"
    kind = "stochastic"
    method = "barostat"
    def __init__(self, mmf, temp, press, start=0, step=1, timecon=1000*femtosecond, anisotropic=True, vol_constraint=False):
        """
        This hook implements the Langevin barostat. The equations are derived in:
            Feller, S. E.; Zhang, Y.; Pastor, R. W.; Brooks, B. R.
            J. Chem. Phys. 1995, 103, 4613-4621

        **ARGUMENTS**
        mmf
            A MicMecForceField instance.
        temp
            The temperature of thermostat.
        press
            The applied pressure for the barostat.

        **OPTIONAL ARGUMENTS**
        start
            The step at which the barostat becomes active.
        timecon
            The time constant of the Langevin barostat.
        anisotropic
            Defines whether anisotropic domain fluctuations are allowed.
        vol_constraint
            Defines whether the volume is allowed to fluctuate.
        """
        self.temp = temp
        self.press = press
        self.timecon = timecon
        self.anisotropic = anisotropic
        self.vol_constraint = vol_constraint
        self.dim = mmf.system.domain.nvec
        # Determine the number of degrees of freedom associated with the unit domain.
        self.baro_ndof = get_ndof_baro(self.dim, self.anisotropic, self.vol_constraint)
        if self.anisotropic:
            # Symmetrize the domain tensor.
            domain_symmetrize(mmf)
        self.domain = mmf.system.domain.rvecs.copy()
        VerletHook.__init__(self, start, step)

    def init(self, iterative):
        self.timestep_press = iterative.timestep
        clean_momenta(iterative.pos, iterative.vel, iterative.masses, iterative.mmf.system.domain)
        # Set the number of internal degrees of freedom (no restriction on p_cm).
        if iterative.ndof is None:
            iterative.ndof = iterative.pos.size
        # Define the barostat "mass".
        self.mass_press = (iterative.ndof+3)/3*boltzmann*self.temp*(self.timecon/(2*np.pi))**2
        # Define initial barostat velocity.
        self.vel_press = get_random_vel_press(self.mass_press, self.temp)
        # Make sure the volume of the domain will not change if applicable.
        if self.vol_constraint:
            self.vel_press -= np.trace(self.vel_press)/3*np.eye(3)
        if not self.anisotropic:
            self.vel_press = self.vel_press[0][0]

        # Compute gpos and vtens, since they differ
        # after symmetrising the domain tensor.
        iterative.gpos[:] = 0.0
        iterative.vtens[:] = 0.0
        iterative.epot = iterative.mmf.compute(iterative.gpos,iterative.vtens)

    def pre(self, iterative, chainvel0 = None):
        # Bookkeeping.
        epot0 = iterative.epot
        ekin0 = iterative.ekin
        # The actual update.
        self.baro(iterative, chainvel0)
        # Some more bookkeeping.
        epot1 = iterative.epot
        ekin1 = iterative.ekin
        self.econs_correction += epot0 - epot1 + ekin0 - ekin1

    def post(self, iterative, chainvel0 = None):
        # Bookkeeping.
        epot0 = iterative.epot
        ekin0 = iterative.ekin
        # The actual update.
        self.baro(iterative, chainvel0)
        # Some more bookkeeping.
        epot1 = iterative.epot
        ekin1 = iterative.ekin
        self.econs_correction += epot0 - epot1 + ekin0 - ekin1

    def baro(self, iterative, chainvel0):
        def update_baro_vel():
            # Updates the barostat velocity tensor.
            # iL h/(8*tau)
            self.vel_press *= np.exp(-self.timestep_press/(8*self.timecon))
            if chainvel0 is not None:
                # iL v_{xi} v_g h/8: extra contribution due to NHC thermostat.
                self.vel_press *= np.exp(-self.timestep_press*chainvel0/8)
            # Definition of P_intV and G.
            ptens_vol = np.dot(iterative.vel.T*iterative.masses, iterative.vel) - iterative.vtens
            ptens_vol = 0.5*(ptens_vol.T + ptens_vol)
            G = (ptens_vol+(2.0*iterative.ekin/iterative.ndof-self.press*iterative.mmf.system.domain.volume)*np.eye(3))/self.mass_press
            R = self.getR()
            if self.vol_constraint:
                G -= np.trace(G)/self.dim*np.eye(self.dim)
                R -= np.trace(R)/self.dim*np.eye(self.dim)
            if not self.anisotropic:
                G = np.trace(G)
                R = R[0][0]
            # iL (G_g-R_p/W) h/4
            self.vel_press += (G-R/self.mass_press)*self.timestep_press/4
            # iL h/(8*tau)
            self.vel_press *= np.exp(-self.timestep_press/(8*self.timecon))
            if chainvel0 is not None:
                # iL v_{xi} v_g h/8: extra contribution due to NHC thermostat.
                self.vel_press *= np.exp(-self.timestep_press*chainvel0/8)

        # First part of the barostat velocity tensor update.
        update_baro_vel()

        # iL v_g h/2
        if self.anisotropic:
            Dr, Qg = np.linalg.eigh(self.vel_press)
            Daccr = np.diagflat(np.exp(Dr*self.timestep_press/2))
            rot_mat = np.dot(np.dot(Qg, Daccr), Qg.T)
            pos_new = np.dot(iterative.pos, rot_mat)
            rvecs_new = np.dot(iterative.rvecs, rot_mat)
        else:
            c = np.exp(self.vel_press*self.timestep_press/2)
            pos_new = c*iterative.pos
            rvecs_new = c*iterative.rvecs

        # Update the positions and domain vectors.
        iterative.mmf.update_pos(pos_new)
        iterative.pos[:] = pos_new
        iterative.mmf.update_rvecs(rvecs_new)
        iterative.rvecs[:] = rvecs_new

        # Update the potential energy.
        iterative.gpos[:] = 0.0
        iterative.vtens[:] = 0.0
        iterative.epot = iterative.mmf.compute(iterative.gpos, iterative.vtens)

        # -iL (v_g + Tr(v_g)/ndof) h/2
        if self.anisotropic:
            if self.vol_constraint:
                Dg, Eg = np.linalg.eigh(self.vel_press)
            else:
                Dg, Eg = np.linalg.eigh(self.vel_press+(np.trace(self.vel_press)/iterative.ndof)*np.eye(3))
            Daccg = np.diagflat(np.exp(-Dg*self.timestep_press/2))
            rot_mat = np.dot(np.dot(Eg, Daccg), Eg.T)
            vel_new = np.dot(iterative.vel, rot_mat)
        else:
            vel_new = np.exp(-((1.0+3.0/iterative.ndof)*self.vel_press)*self.timestep_press/2) * iterative.vel
        iterative.vel[:] = vel_new

        # Update kinetic energy.
        iterative.ekin = iterative._compute_ekin()

        # Second part of the barostat velocity tensor update.
        update_baro_vel()

    def getR(self):
        shape = 3, 3
        # Generate random 3x3 tensor.
        rand = np.random.normal(0, 1, shape)*np.sqrt(2*self.mass_press*boltzmann*self.temp/(self.timestep_press*self.timecon))
        R = np.zeros(shape)
        # Create initial symmetric pressure velocity tensor.
        for i in range(3):
            for j in range(3):
                if i >= j:
                    R[i,j] = rand[i,j]
                else:
                    R[i,j] = rand[j,i]
        return R


class MTKBarostat(VerletHook):
    name = "MTTK"
    kind = "deterministic"
    method = "barostat"
    def __init__(self, mmf, temp, press, start=0, step=1, timecon=1000*femtosecond, anisotropic=True, vol_constraint=False, baro_thermo=None, vel_press0=None, restart=False):
        """
        This hook implements the Martyna-Tobias-Klein barostat. The equations
        are derived in:
            Martyna, G. J.; Tobias, D. J.; Klein, M. L. J. Chem. Phys. 1994,
            101, 4177-4189.
        The implementation (used here) of a symplectic integrator of this
        barostat is discussed in
            Martyna, G. J.;  Tuckerman, M. E.;  Tobias, D. J.;  Klein,
            M. L. Mol. Phys. 1996, 87, 1117-1157.
        
        **ARGUMENTS**
        mmf
            A MicMecForceField instance.
        temp
            The temperature of thermostat.
        press
            The applied pressure for the barostat.
        
        **OPTIONAL ARGUMENTS**
        start
            The step at which the barostat becomes active.
        timecon
            The time constant of the Martyna-Tobias-Klein barostat.
        anisotropic
            Defines whether anisotropic domain fluctuations are allowed.
        vol_constraint
            Defines whether the volume is allowed to fluctuate.
        baro_thermo
            NHCThermostat instance, coupled directly to the barostat.
        vel_press0
            The initial barostat velocity tensor.
        restart
            If true, the domain is not symmetrized initially.
        """
        self.temp = temp
        self.press = press
        self.timecon_press = timecon
        self.anisotropic = anisotropic
        self.vol_constraint = vol_constraint
        self.baro_thermo = baro_thermo
        self.dim = mmf.system.domain.nvec
        self.restart = restart
        # Determine the number of degrees of freedom associated with the unit domain.
        self.baro_ndof = get_ndof_baro(self.dim, self.anisotropic, self.vol_constraint)
        if self.anisotropic and not self.restart:
            # Symmetrize the domain tensor.
            domain_symmetrize(mmf)
        self.domain = mmf.system.domain.rvecs.copy()
        self.vel_press = vel_press0
        VerletHook.__init__(self, start, step)

    def init(self, iterative):
        self.timestep_press = iterative.timestep
        if not self.restart:
            clean_momenta(iterative.pos, iterative.vel, iterative.masses, iterative.mmf.system.domain)
        # Determine the internal degrees of freedom.
        if iterative.ndof is None:
            iterative.ndof = get_ndof_internal_md(iterative.mmf.system.nnodes, iterative.mmf.system.domain.nvec)
        # Determine barostat "mass".
        angfreq = 2*np.pi/self.timecon_press
        self.mass_press = (iterative.ndof + self.dim**2)*boltzmann*self.temp/angfreq**2
        if self.vel_press is None:
            # Define initial barostat velocity.
            self.vel_press = get_random_vel_press(self.mass_press, self.temp)
            if not self.anisotropic:
                self.vel_press = self.vel_press[0][0]
        # Initialize the barostat thermostat if present.
        if self.baro_thermo is not None:
            self.baro_thermo.chain.timestep = iterative.timestep
            self.baro_thermo.chain.set_ndof(self.baro_ndof)
        # Make sure the volume of the domain will not change if applicable.
        if self.vol_constraint:
            self.vel_press -= np.trace(self.vel_press)/3*np.eye(3)
        # Compute gpos and vtens, since they differ
        # after symmetrising the domain tensor.
        iterative.gpos[:] = 0.0
        iterative.vtens[:] = 0.0
        iterative.epot = iterative.mmf.compute(iterative.gpos, iterative.vtens)

    def pre(self, iterative, chainvel0 = None):
        if self.baro_thermo is not None:
            # Overrule the chainvel0 argument in the TBC instance.
            chainvel0 = self.baro_thermo.chain.vel[0]
        # Propagate the barostat.
        self.baro(iterative, chainvel0)
        # Propagate the barostat thermostat if present.
        if self.baro_thermo is not None:
            # Determine the barostat kinetic energy.
            ekin_baro = self._compute_ekin_baro()
            # Update the barostat thermostat, without updating v_g (done in self.baro).
            vel_press_copy = np.zeros(self.vel_press.shape)
            vel_press_copy[:] = self.vel_press
            dummy_vel, dummy_ekin = self.baro_thermo.chain(ekin_baro, vel_press_copy, 0)

    def post(self, iterative, chainvel0 = None):
        # Propagate the barostat thermostat if present.
        if self.baro_thermo is not None:
            # Determine the barostat kinetic energy.
            ekin_baro = self._compute_ekin_baro()
            # Update the barostat thermostat, without updating v_g (done in self.baro).
            vel_press_copy = np.zeros(self.vel_press.shape)
            vel_press_copy[:] = self.vel_press
            dummy_vel, dummy_ekin = self.baro_thermo.chain(ekin_baro, vel_press_copy, 0)
            # Overrule the chainvel0 argument in the TBC instance.
            chainvel0 = self.baro_thermo.chain.vel[0]
        # Propagate the barostat.
        self.baro(iterative, chainvel0)
        # Calculate the correction due to the barostat alone.
        self.econs_correction = self._compute_ekin_baro()
        # Add the PV term if the volume is not constrained.
        if not self.vol_constraint:
            self.econs_correction += self.press*iterative.mmf.system.domain.volume
        if self.baro_thermo is not None:
            # Add the correction due to the barostat thermostat.
            self.econs_correction += self.baro_thermo.chain.get_econs_correction()

    def baro(self, iterative, chainvel0):
        def update_baro_vel():
            # Updates the barostat velocity tensor.
            if chainvel0 is not None:
                # iL v_{xi} v_g h/8
                self.vel_press *= np.exp(-self.timestep_press*chainvel0/8)
            # Definition of P_intV and G.
            ptens_vol = np.dot(iterative.vel.T*iterative.masses, iterative.vel) - iterative.vtens
            ptens_vol = 0.5*(ptens_vol.T + ptens_vol)
            G = (ptens_vol+(2.0*iterative.ekin/iterative.ndof-self.press*iterative.mmf.system.domain.volume)*np.eye(3))/self.mass_press
            if not self.anisotropic:
                G = np.trace(G)
            if self.vol_constraint:
                G -= np.trace(G)/self.dim*np.eye(self.dim)
            # iL G_g h/4
            self.vel_press += G*self.timestep_press/4
            if chainvel0 is not None:
                # iL v_{xi} v_g h/8
                self.vel_press *= np.exp(-self.timestep_press*chainvel0/8)

        # First part of the barostat velocity tensor update.
        update_baro_vel()

        # iL v_g h/2
        if self.anisotropic:
            Dr, Qg = np.linalg.eigh(self.vel_press)
            Daccr = np.diagflat(np.exp(Dr*self.timestep_press/2))
            rot_mat = np.dot(np.dot(Qg, Daccr), Qg.T)
            pos_new = np.dot(iterative.pos, rot_mat)
            rvecs_new = np.dot(iterative.rvecs, rot_mat)
        else:
            c = np.exp(self.vel_press*self.timestep_press/2)
            pos_new = c*iterative.pos
            rvecs_new = c*iterative.rvecs

        # Update the positions and domain vectors.
        iterative.mmf.update_pos(pos_new)
        iterative.pos[:] = pos_new
        iterative.mmf.update_rvecs(rvecs_new)
        iterative.rvecs[:] = rvecs_new

        # Update the potential energy.
        iterative.gpos[:] = 0.0
        iterative.vtens[:] = 0.0
        iterative.epot = iterative.mmf.compute(iterative.gpos, iterative.vtens)

        # -iL (v_g + Tr(v_g)/ndof) h/2
        if self.anisotropic:
            if self.vol_constraint:
                Dg, Eg = np.linalg.eigh(self.vel_press)
            else:
                Dg, Eg = np.linalg.eigh(self.vel_press+(np.trace(self.vel_press)/iterative.ndof)*np.eye(3))
            Daccg = np.diagflat(np.exp(-Dg*self.timestep_press/2))
            rot_mat = np.dot(np.dot(Eg, Daccg), Eg.T)
            vel_new = np.dot(iterative.vel, rot_mat)
        else:
            vel_new = np.exp(-((1.0 + 3.0/iterative.ndof)*self.vel_press)*self.timestep_press/2)*iterative.vel

        # Update the velocities and the kinetic energy.
        iterative.vel[:] = vel_new
        iterative.ekin = iterative._compute_ekin()

        # Second part of the barostat velocity tensor update.
        update_baro_vel()

    def add_press_cont(self):
        kt = self.temp*boltzmann
        # Pressure contribution to thermostat: kinetic domain tensor energy
        # and extra degrees of freedom due to domain tensor.
        if self.baro_thermo is None:
            return 2*self._compute_ekin_baro() - self.baro_ndof*kt
        else:
            # If a barostat thermostat is present, the thermostat is decoupled.
            return 0

    def _compute_ekin_baro(self):
        # Returns the kinetic energy associated with the domain fluctuations.
        if self.anisotropic:
            return 0.5*self.mass_press*np.trace(np.dot(self.vel_press.T,self.vel_press))
        else:
            return 0.5*self.mass_press*self.vel_press**2

class PRBarostat(VerletHook):
    name = "PR"
    kind = "deterministic"
    method = "barostat"
    def __init__(self, mmf, temp, press, start=0, step=1, timecon=1000*femtosecond, anisotropic=True, vol_constraint=False, baro_thermo=None, vel_press0=None, restart=False):
        """
        This hook implements the Parrinello-Rahman barostat for finite strains.
        The equations are derived in:
            Parrinello, M.; Rahman, A. J. Appl. Phys. 1981, 52, 7182-7190.
        
        **ARGUMENTS**
        mmf
            A MicMecForceField instance.
        temp
            The temperature of thermostat.
        press
            The applied pressure for the barostat.
        
        **OPTIONAL ARGUMENTS**
        start
            The step at which the barostat becomes active.
        timecon
            The time constant of the Tadmor-Miller barostat.
        anisotropic
            Defines whether anisotropic domain fluctuations are allowed.
        vol_constraint
            Defines whether the volume is allowed to fluctuate.
        baro_thermo
            NHCThermostat instance, coupled directly to the barostat.
        vel_press0
            The initial barostat velocity tensor.
        restart
            If true, the domain is not symmetrized initially.
        """

        self.temp = temp
        if isinstance(press, np.ndarray):
            self.press = press
        else:
            self.press = press*np.eye(3)
        self.timecon_press = timecon
        self.anisotropic = anisotropic
        self.vol_constraint = vol_constraint
        self.baro_thermo = baro_thermo
        self.dim = mmf.system.domain.nvec
        self.restart = restart
        # Determine the number of degrees of freedom associated with the unit domain.
        self.baro_ndof = get_ndof_baro(self.dim, self.anisotropic, self.vol_constraint)
        if self.anisotropic and not self.restart:
            # Symmetrize the domain and stress tensor.
            vc, tn = domain_symmetrize(mmf, tensor_list=[self.press])
            self.press = tn[0]
        self.P = np.trace(self.press)/3
        self.S_ani = self.press - self.P*np.eye(3)
        # Only symmetric part of stress tensor contributes to individual motion.
        self.S_ani = 0.5*(self.S_ani + self.S_ani.T)
        self.domaininv0 = np.linalg.inv(mmf.system.domain.rvecs.copy())
        self.vol0 = mmf.system.domain.volume
        # Definition of Sigma = V_0*h_0^{-1} S_ani h_0^{-T}
        self.Sigma = self.vol0*np.dot(np.dot(self.domaininv0.T, self.S_ani), self.domaininv0)
        self.vel_press = vel_press0
        VerletHook.__init__(self, start, step)

    def init(self, iterative):
        self.timestep_press = iterative.timestep
        if not self.restart:
            clean_momenta(iterative.pos, iterative.vel, iterative.masses, iterative.mmf.system.domain)
        # Determine the internal degrees of freedom.
        if iterative.ndof is None:
            iterative.ndof = get_ndof_internal_md(iterative.mmf.system.nnodes, iterative.mmf.system.domain.nvec)
        # Determine barostat "mass" following W = timecon*np.sqrt(n_part) * m_av
        # n_part = len(iterative.masses)
        # self.mass_press = self.timecon_press*np.sum(iterative.masses)/np.sqrt(iterative.mmf.system.nnodes)
        angfreq = 2*np.pi/self.timecon_press
        self.mass_press = (iterative.ndof+self.dim**2)*boltzmann*self.temp/angfreq**2/self.vol0**(2./3)
        if self.vel_press is None:
            # Define initial barostat velocity.
            self.vel_press = get_random_vel_press(self.mass_press, self.temp)
            if not self.anisotropic:
                self.vel_press = self.vel_press[0][0]
        # Initialize the barostat thermostat if present.
        if self.baro_thermo is not None:
            self.baro_thermo.chain.timestep = iterative.timestep
            self.baro_thermo.chain.set_ndof(self.baro_ndof)
        # Make sure the volume of the domain will not change if applicable.
        if self.vol_constraint:
            self.vel_press -= np.trace(self.vel_press)/3*np.eye(3)
        # Compute gpos and vtens, since they differ
        # after symmetrising the domain tensor.
        iterative.gpos[:] = 0.0
        iterative.vtens[:] = 0.0
        iterative.epot = iterative.mmf.compute(iterative.gpos, iterative.vtens)

    def pre(self, iterative, chainvel0 = None):
        if self.baro_thermo is not None:
            # Overrule the chainvel0 argument in the TBC instance.
            chainvel0 = self.baro_thermo.chain.vel[0]
        # Propagate the barostat.
        self.baro(iterative, chainvel0)
        # Propagate the barostat thermostat if present.
        if self.baro_thermo is not None:
            # Determine the barostat kinetic energy.
            ekin_baro = self._compute_ekin_baro()
            # Update the barostat thermostat, without updating v_g (done in self.baro).
            vel_press_copy = np.zeros(self.vel_press.shape)
            vel_press_copy[:] = self.vel_press
            dummy_vel, dummy_ekin = self.baro_thermo.chain(ekin_baro, vel_press_copy, 0)

    def post(self, iterative, chainvel0 = None):
        # Propagate the barostat thermostat if present.
        if self.baro_thermo is not None:
            # Determine the barostat kinetic energy.
            ekin_baro = self._compute_ekin_baro()
            # Update the barostat thermostat, without updating v_g (done in self.baro).
            vel_press_copy = np.zeros(self.vel_press.shape)
            vel_press_copy[:] = self.vel_press
            dummy_vel, dummy_ekin = self.baro_thermo.chain(ekin_baro, vel_press_copy, 0)
            # Overrule the chainvel0 argument in the TBC instance.
            chainvel0 = self.baro_thermo.chain.vel[0]
        # Propagate the barostat.
        self.baro(iterative, chainvel0)
        # Calculate the correction due to the barostat alone.
        h = iterative.rvecs.copy()
        self.econs_correction = self._compute_ekin_baro() + self.P*iterative.mmf.system.domain.volume + 0.5*np.trace(np.dot(np.dot(self.Sigma, h), h.T))

        if self.baro_thermo is not None:
            # Add the correction due to the barostat thermostat.
            self.econs_correction += self.baro_thermo.chain.get_econs_correction()

    def baro(self, iterative, chainvel0):
        def update_baro_vel():
            # Updates the barostat velocity tensor.
            if chainvel0 is not None:
                # iL v_{xi} v_g h/8
                self.vel_press *= np.exp(-self.timestep_press*chainvel0/8)
            h = iterative.rvecs.copy()
            ptens_vol = np.dot(iterative.vel.T*iterative.masses, iterative.vel) - iterative.vtens
            S_PK2_vol = np.dot(np.dot(h.T, self.Sigma), h)
            # Definition of G.
            G = np.dot(ptens_vol-self.P*iterative.mmf.system.domain.volume*np.eye(3)-S_PK2_vol, np.linalg.inv(h))/self.mass_press
            G = 0.5*(G+G.T)
            # G = G+G.T-np.diagflat(np.diag(G))
            if not self.anisotropic:
                G = np.trace(G)
            if self.vol_constraint:
                G -= np.trace(G)/self.dim*np.eye(self.dim)
            # iL G_h h/4
            self.vel_press += G*self.timestep_press/4
            if chainvel0 is not None:
                # iL v_{xi} v_g h/8
                self.vel_press *= np.exp(-self.timestep_press*chainvel0/8)

        # First part of the barostat velocity tensor update.
        update_baro_vel()

        # Update of the positions and domain vectors.
        rvecs_new = iterative.rvecs + self.vel_press*self.timestep_press/2

        if self.anisotropic:
            Dr, Qg = np.linalg.eigh(np.dot(self.vel_press, np.linalg.inv(iterative.rvecs.T)))
        else:
            Dr, Qg = np.linalg.eigh(self.vel_press*np.linalg.inv(iterative.rvecs.T))
        Daccr = np.diagflat(np.exp(Dr*self.timestep_press/2))
        Daccv = np.diagflat(np.exp(-Dr*self.timestep_press/2))
        rot_mat_r = np.dot(np.dot(Qg, Daccr), Qg.T)
        rot_mat_v = np.dot(np.dot(Qg, Daccv), Qg.T)
        pos_new = np.dot(iterative.pos, rot_mat_r)
        vel_new = np.dot(iterative.vel, rot_mat_v)

        iterative.mmf.update_pos(pos_new)
        iterative.pos[:] = pos_new
        iterative.mmf.update_rvecs(rvecs_new)
        iterative.rvecs[:] = rvecs_new
        iterative.vel[:] = vel_new

        # Update the potential and kinetic energy.
        iterative.gpos[:] = 0.0
        iterative.vtens[:] = 0.0
        iterative.epot = iterative.mmf.compute(iterative.gpos, iterative.vtens)
        iterative.ekin = iterative._compute_ekin()

        # Second part of the barostat velocity tensor update.
        update_baro_vel()

    def add_press_cont(self):
        kt = self.temp*boltzmann
        # Pressure contribution to thermostat: kinetic domain tensor energy
        # and extra degrees of freedom due to domain tensor.
        if self.baro_thermo is None:
            return 2*self._compute_ekin_baro() - self.baro_ndof*kt
        else:
            # If a barostat thermostat is present, the thermostat is decoupled.
            return 0

    def _compute_ekin_baro(self):
        # Returns the kinetic energy associated with the domain fluctuations.
        if self.anisotropic:
            return 0.5*self.mass_press*np.trace(np.dot(self.vel_press.T,self.vel_press))
        else:
            return 0.5*self.mass_press*self.vel_press**2


class TadmorBarostat(VerletHook):
    name = "Tadmor"
    kind = "deterministic"
    method = "barostat"
    def __init__(self, mmf, temp, press, start=0, step=1, timecon=1000*femtosecond, anisotropic=True, vol_constraint=False, baro_thermo=None, vel_press0=None, restart=False):
        """
        This hook implements the Tadmor-Miller barostat for finite strains.
        The equations are derived in:
            Tadmor, E. B.; Miller, R. E. Modeling Materials: Continuum,
            Atomistic and Multiscale Techniques 2011, 520-527.
        
        **ARGUMENTS**
        mmf
            A MicMecForceField instance.
        temp
            The temperature of thermostat.
        press
            The applied second Piola-Kirchhoff tensor for the barostat.
            
        **OPTIONAL ARGUMENTS**
        start
            The step at which the barostat becomes active.
        timecon
            The time constant of the Tadmor-Miller barostat.
        anisotropic
            Defines whether anisotropic domain fluctuations are allowed.
        vol_constraint
            Defines whether the volume is allowed to fluctuate.
        baro_thermo
            NHCThermostat instance, coupled directly to the barostat.
        vel_press0
            The initial barostat velocity tensor.
        restart
            If true, the domain is not symmetrized initially.
        """
        self.temp = temp
        self.press = press
        self.timecon_press = timecon
        self.anisotropic = anisotropic
        self.vol_constraint = vol_constraint
        self.baro_thermo = baro_thermo
        self.dim = mmf.system.domain.nvec
        self.restart = restart
        # Determine the number of degrees of freedom associated with the unit domain.
        self.baro_ndof = get_ndof_baro(self.dim, self.anisotropic, self.vol_constraint)
        if self.anisotropic and not self.restart:
            # Symmetrize the domain tensor.
            domain_symmetrize(mmf)
        self.domain = mmf.system.domain.rvecs.copy()
        self.domaininv0 = np.linalg.inv(self.domain)
        self.vol0 = mmf.system.domain.volume
        # Definition of h0^{-1} S h_0^{-T}.
        self.Strans = np.dot(np.dot(self.domaininv0, self.press), self.domaininv0.T)
        self.vel_press = vel_press0
        VerletHook.__init__(self, start, step)

    def init(self, iterative):
        self.timestep_press = iterative.timestep
        if not self.restart:
            clean_momenta(iterative.pos, iterative.vel, iterative.masses, iterative.mmf.system.domain)
        # Determine the internal degrees of freedom.
        if iterative.ndof is None:
            iterative.ndof = get_ndof_internal_md(iterative.mmf.system.nnodes, iterative.mmf.system.domain.nvec)
        # Determine barostat "mass".
        angfreq = 2*np.pi/self.timecon_press
        self.mass_press = (iterative.ndof+self.dim**2)*boltzmann*self.temp/angfreq**2
        if self.vel_press is None:
            # Define initial barostat velocity.
            self.vel_press = get_random_vel_press(self.mass_press, self.temp)
            if not self.anisotropic:
                self.vel_press = self.vel_press[0][0]
        # Initialize the barostat thermostat if present.
        if self.baro_thermo is not None:
            self.baro_thermo.chain.timestep = iterative.timestep
            self.baro_thermo.chain.set_ndof(self.baro_ndof)
        # Make sure the volume of the domain will not change if applicable.
        if self.vol_constraint:
            self.vel_press -= np.trace(self.vel_press)/3*np.eye(3)
        # Compute gpos and vtens, since they differ
        # after symmetrising the domain tensor.
        iterative.gpos[:] = 0.0
        iterative.vtens[:] = 0.0
        iterative.epot = iterative.mmf.compute(iterative.gpos, iterative.vtens)

    def pre(self, iterative, chainvel0 = None):
        if self.baro_thermo is not None:
            # Overrule the chainvel0 argument in the TBC instance.
            chainvel0 = self.baro_thermo.chain.vel[0]
        # Propagate the barostat.
        self.baro(iterative, chainvel0)
        # Propagate the barostat thermostat if present.
        if self.baro_thermo is not None:
            # Determine the barostat kinetic energy.
            ekin_baro = self._compute_ekin_baro()
            # Update the barostat thermostat, without updating v_g (done in self.baro).
            vel_press_copy = np.zeros(self.vel_press.shape)
            vel_press_copy[:] = self.vel_press
            dummy_vel, dummy_ekin = self.baro_thermo.chain(ekin_baro, vel_press_copy, 0)

    def post(self, iterative, chainvel0 = None):
        # Propagate the barostat thermostat if present.
        if self.baro_thermo is not None:
            # Determine the barostat kinetic energy.
            ekin_baro = self._compute_ekin_baro()
            # Update the barostat thermostat, without updating v_g (done in self.baro).
            vel_press_copy = np.zeros(self.vel_press.shape)
            vel_press_copy[:] = self.vel_press
            dummy_vel, dummy_ekin = self.baro_thermo.chain(ekin_baro, vel_press_copy, 0)
            # Overrule the chainvel0 argument in the TBC instance.
            chainvel0 = self.baro_thermo.chain.vel[0]
        # Propagate the barostat.
        self.baro(iterative, chainvel0)
        # Calculate the Lagrangian strain.
        eta = 0.5*(np.dot(self.domaininv0.T,np.dot(iterative.rvecs.T,np.dot(iterative.rvecs,self.domaininv0)))-np.eye(3))
        # Calculate the correction due to the barostat alone.
        ang_ten = self._compute_angular_tensor(iterative.pos, iterative.vel, iterative.masses)
        self.econs_correction = self.vol0*np.trace(np.dot(self.press.T,eta)) + self._compute_ekin_baro() - np.trace(np.dot(ang_ten, self.vel_press))
        if self.baro_thermo is not None:
            # Add the correction due to the barostat thermostat.
            self.econs_correction += self.baro_thermo.chain.get_econs_correction()

    def baro(self, iterative, chainvel0):
        def update_baro_vel():
            # Updates the barostat velocity tensor.
            if chainvel0 is not None:
                # iL v_{xi} v_g h/8
                self.vel_press *= np.exp(-self.timestep_press*chainvel0/8)
            # Definition of sigma_{int} V.
            ptens_vol = np.dot(iterative.vel.T*iterative.masses, iterative.vel) - iterative.vtens
            ptens_vol = 0.5*(ptens_vol.T + ptens_vol)
            # Definition of (Delta sigma) V.
            ang_ten = self._compute_angular_tensor(iterative.pos, iterative.vel, iterative.masses)
            dptens_vol = np.dot(self.vel_press.T, ang_ten.T) - np.dot(ang_ten.T, self.vel_press.T)
            # Definition of \tilde{sigma} V.
            sigmaS_vol = self.vol0*np.dot(np.dot(iterative.rvecs,self.Strans), iterative.rvecs.T)
            # Definition of G.
            G = (ptens_vol - sigmaS_vol + dptens_vol + 2.0*iterative.ekin/iterative.ndof*np.eye(3))/self.mass_press
            if not self.anisotropic:
                G = np.trace(G)
            if self.vol_constraint:
                G -= np.trace(G)/self.dim*np.eye(self.dim)
            # iL G_g h/4
            self.vel_press += G*self.timestep_press/4
            if chainvel0 is not None:
                # iL v_{xi} v_g h/8
                self.vel_press *= np.exp(-self.timestep_press*chainvel0/8)

        # First part of the barostat velocity tensor update.
        update_baro_vel()

        # iL v_g h/2
        if self.anisotropic:
            Dr, Qg = np.linalg.eigh(self.vel_press)
            Daccr = np.diagflat(np.exp(Dr*self.timestep_press/2))
            rot_mat = np.dot(np.dot(Qg, Daccr), Qg.T)
            rvecs_new = np.dot(iterative.rvecs, rot_mat)
        else:
            c = np.exp(self.vel_press*self.timestep_press/2)
            rvecs_new = c*iterative.rvecs

        # Update the positions and domain vectors.
        iterative.mmf.update_rvecs(rvecs_new)
        iterative.rvecs[:] = rvecs_new

        # Update the potential energy.
        iterative.gpos[:] = 0.0
        iterative.vtens[:] = 0.0
        iterative.epot = iterative.mmf.compute(iterative.gpos, iterative.vtens)

        # -iL (Tr(v_g)/ndof) h/2
        if self.anisotropic:
            if not self.vol_constraint:
                vel_new = np.exp(-np.trace(self.vel_press)/iterative.ndof*self.timestep_press/2)
        else:
            vel_new = np.exp(-3.0/iterative.ndof*self.vel_press*self.timestep_press/2) * iterative.vel

        # Update the velocities and the kinetic energy.
        iterative.vel[:] = vel_new
        iterative.ekin = iterative._compute_ekin()

        # Second part of the barostat velocity tensor update.
        update_baro_vel()

    def add_press_cont(self):
        kt = self.temp*boltzmann
        # Pressure contribution to thermostat: kinetic domain tensor energy
        # and extra degrees of freedom due to domain tensor.
        if self.baro_thermo is None:
            return 2*self._compute_ekin_baro() - self.baro_ndof*kt
        else:
            # If a barostat thermostat is present, the thermostat is decoupled.
            return 0

    def _compute_ekin_baro(self):
        # Returns the kinetic energy associated with the domain fluctuations.
        if self.anisotropic:
            return 0.5*self.mass_press*np.trace(np.dot(self.vel_press.T,self.vel_press))
        else:
            return 0.5*self.mass_press*self.vel_press**2

    def _compute_angular_tensor(self, pos, vel, masses):
        return np.dot(pos.T, masses.reshape(-1,1)*vel)


class MTKAttributeStateItem(StateItem):
    def __init__(self, attr):
        StateItem.__init__(self, "baro_"+attr)
        self.attr = attr

    def get_value(self, iterative):
        baro = None
        for hook in iterative.hooks:
            if isinstance(hook, MTKBarostat):
                baro = hook
                break
            elif isinstance(hook, TBCombination):
                if isinstance(hook.barostat, MTKBarostat):
                    baro = hook.barostat
                break
        if baro is None:
            raise TypeError("Iterative does not contain an MTKBarostat hook.")
        if self.key.startswith("baro_chain_"):
            if baro.baro_thermo is not None:
                key = self.key.split("_")[2]
                return getattr(baro.baro_thermo.chain, key)
            else: return 0
        else: return getattr(baro, self.attr)

    def copy(self):
        return self.__class__(self.attr)


