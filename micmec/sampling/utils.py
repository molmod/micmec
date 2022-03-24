#!/usr/bin/env python
# File name: utils.py
# Description: Useful functions for use in a micromechanical simulation.
# Author: Joachim Vandewalle
# Date: 26-10-2021

from molmod import boltzmann

import numpy as np

def get_random_vel(temp0, scalevel0, masses, select=None):
    """
    Generate random nodal velocities using a Maxwell-Boltzmann distribution.
    
    **ARGUMENTS**
    temp0
        The temperature for the Maxwell-Boltzmann distribution.
    scalevel0
        When set to True, the velocities are rescaled such that the
        instantaneous temperature coincides with temp0.
    masses
        An (N,) array with nodal masses.
    
    **OPTIONAL ARGUMENTS**
    select
        When given, this must be an array of integer indexes. Only for these
        nodes (masses) initial velocities will be generated.

    **RETURNS** 
    An (N, 3) array with random nodal velocities. When the select option is used,
    the shape of the results is (M, 3), where M is the length of the selected array.
    """
    if select is not None:
        masses = masses[select]
    shapevel0 = (len(masses), 3)
    vel0 = np.random.normal(0, 1, shapevel0)*np.sqrt(boltzmann*temp0/masses).reshape(-1,1)
    if scalevel0 and temp0 > 0:
        temp = np.mean((vel0**2*masses.reshape(-1,1)))/boltzmann
        scale = np.sqrt(temp0/temp)
        vel0 *= scale
    return vel0


def remove_com_moment(vel, masses):
    """
    Zero the global linear momentum.

    **ARGUMENTS**
    vel
        An (N, 3) array with nodal velocities. This array is modified in-place.
    masses
        An (N,) array with nodal masses.
    
    The zero linear center-of-mass momentum is achieved by subtracting 
    translational rigid body motion from the nodal velocities.
    """
    # Compute the center of mass velocity.
    velcom = np.dot(masses, vel)/np.sum(masses)
    # Subtract this com velocity vector from each node velocity.
    vel[:] -= velcom


def remove_angular_moment(pos, vel, masses):
    """
    Zero the global angular momentum.

    **ARGUMENTS**
    pos
        An (N, 3) array with nodal positions. This array is not modified.
    vel
        An (N, 3) array with nodal velocities. This array is modified in-place.
    masses
        An (N,) array with nodal masses.
    
    The zero angular momentum is achieved by subtracting angular rigid body motion 
    from the nodal velocities. The angular momentum is measured with respect to the 
    center of mass to avoid that this routine reintroduces a linear center-of-mass 
    velocity. This is also beneficial for the numerical stability.
    """
    # Translate a copy of the positions, such that the center of mass lies in the origin.
    pos = pos.copy()
    poscom = np.dot(masses, pos)/np.sum(masses)
    pos -= poscom
    # Compute the inertia tensor.
    itens = inertia_tensor(pos, masses)
    # Compute the angular momentum vector.
    amom = angular_moment(pos, vel, masses)
    # Compute the angular velocity vector.
    avel = angular_velocity(amom, itens)
    # Subtract the rigid body angular velocities from the nodal velocities.
    vel[:] -= rigid_body_angular_velocities(pos, avel)


def clean_momenta(pos, vel, masses, domain):
    """
    Remove any relevant external momenta.
    
    **ARGUMENTS**
    pos
        An (N, 3) array with nodal positions. This array is not modified.
    vel
        An (N, 3) array with nodal velocities. This array is modified in-place.
    masses
        An (N,) array with nodal masses.
    domain
        An instance of a simulation domain describing the periodic boundary conditions.
    """
    remove_com_moment(vel, masses)
    if domain.nvec == 0:
        # Remove all angular momenta.
        remove_angular_moment(pos, vel, masses)
    elif domain.nvec == 1:
        # TODO: only the angular momentum about the domain vector has to be projected out.
        raise NotImplementedError


def inertia_tensor(pos, masses):
    """
    Compute the inertia tensor for a given set of point particles.
    
    **ARGUMENTS**
    pos
        An (N, 3) array with nodal positions.
    masses
        An (N,) array with nodal masses.
    
    **RETURNS** 
    A (3, 3) array containing the inertia tensor.
    """
    return np.identity(3)*(masses.reshape(-1,1)*pos**2).sum() - np.dot(pos.T, masses.reshape(-1,1)*pos)


def angular_moment(pos, vel, masses):
    """
    Compute the angular moment of a set of point particles.
        
    **ARGUMENTS**
    pos
        An (N, 3) array with atomic positions.
    vel
        An (N, 3) array with atomic velocities.
    masses
        An (N,) array with atomic masses.
        
    **RETURNS** 
    A (3,) array with the angular momentum vector.
    """
    lin_moms = masses.reshape(-1,1)*vel
    ang_mom = np.zeros(3, float)
    ang_mom[0] = np.sum(pos[:,1]*lin_moms[:,2] - pos[:,2]*lin_moms[:,1])
    ang_mom[1] = np.sum(pos[:,2]*lin_moms[:,0] - pos[:,0]*lin_moms[:,2])
    ang_mom[2] = np.sum(pos[:,0]*lin_moms[:,1] - pos[:,1]*lin_moms[:,0])
    return ang_mom


def angular_velocity(amom, itens, epsilon=1e-10):
    """
    Derive the angular velocity from the angular moment and the inertia tensor.
    
    **ARGUMENTS**
    amom
        An (3,) array with angular momenta.
    itens
        A (3, 3) array with the inertia tensor.
    
    **OPTIONAL ARGUMENTS**
    epsilon
        A threshold for the low eigenvalues of the inertia tensor. When an
        eigenvalue is below this threshold, it is assumed to be zero plus
        some (irrelevant) numerical noise.
    
    **RETURNS** 
    A (3,) array with the angular velocity vector.
    In principle this routine should merely return:
        np.linalg.solve(itens, amom).
    However, when the inertia tensor has zero eigenvalues, this routine will 
    use a proper pseudo-inverse of the inertia tensor.
    """
    evals, evecs = np.linalg.eigh(itens)
    # Select the significant part of the decomposition.
    mask = evals > epsilon
    evals = evals[mask]
    evecs = evecs[:, mask]
    # Compute the pseudoinverse.
    return np.dot(evecs, np.dot(evecs.T, amom)/evals)


def rigid_body_angular_velocities(pos, avel):
    """
    Generate the velocities of a set of nodes that move as a rigid body.

    **ARGUMENTS**
    pos
        An (N, 3) array with nodal positions.
    avel
        An (3,) array with the angular velocity vector of the rigid body.
    
    **RETURNS** 
    An (N, 3) array with nodal velocities in the rigid body.
    The linear momentum of the rigid body is zero.
    """
    vel = np.zeros(pos.shape, float)
    vel[:,0] = (avel[1]*pos[:,2] - avel[2]*pos[:,1])
    vel[:,1] = (avel[2]*pos[:,0] - avel[0]*pos[:,2])
    vel[:,2] = (avel[0]*pos[:,1] - avel[1]*pos[:,0])
    return vel


def get_ndof_internal_md(nnodes, nper):
    """
    Return the effective number of internal degrees of freedom for MD simulations.

    **ARGUMENTS**
    nnodes
        The number of nodes.
    nper
        The number of periodic boundary conditions (0 for isolated systems).
    """
    if nper == 0:
        # There are at least eight nodes in a micromechanical system.
        return 3*nnodes - 6
    elif nper == 1:
        # 1D periodic.
        # Three translations and one rotation about the domain vector.
        return 3*nnodes - 4
    else:
        # 2D and 3D periodic.
        # Three translations.
        return 3*nnodes - 3


def domain_symmetrize(mmf, vector_lst = None, tensor_lst = None):
    """
    Symmetrizes the simulation domain tensor, and updates the position vectors.
    
    **ARGUMENTS**
    mmf
        A MicMecForceField instance.
    
    **OPTIONAL ARGUMENTS**
    vector_lst
        A list of numpy vectors which should be transformed under the symmetrization. 
        Note that the positions are already transformed automatically.
    tensor_lst
        A list of numpy tensors of rank 2 which should be transformed under the symmetrization.
    """
    # Store the unit domain tensor.
    domain = mmf.system.domain.rvecs.copy()
    # SVD decomposition of domain tensor.
    U, s, Vt = np.linalg.svd(domain)
    # Definition of the rotation matrix to symmetrize domain tensor.
    rot_mat = np.dot(Vt.T, U.T)
    # Symmetrize domain tensor and update domain.
    domain = np.dot(domain, rot_mat)
    mmf.update_rvecs(domain)
    # Also update the new nodal positions.
    posnew = np.dot(mmf.system.pos, rot_mat)
    mmf.update_pos(posnew)
    # Initialize the new vector and tensor lists.
    new_vector_lst = []
    new_tensor_lst = []
    # Update the additional vectors from vector_lst.
    if vector_lst is not None:
        for i in range(len(vector_lst)):
            new_vector_lst.append(np.dot(vector_lst[i], rot_mat))
    # Update the additional tensors from tensor_lst.
    if tensor_lst is not None:
        for i in range(len(tensor_lst)):
            new_tensor_lst.append(np.dot(np.dot(rot_mat.T, tensor_lst[i]), rot_mat))
    return new_vector_lst, new_tensor_lst


def domain_lower(rvecs):
    """
    Transform the simulation domain matrix to its lower diagonal form. The transformation
    is described here https://lammps.sandia.gov/doc/Howto_triclinic.html,
    bearing in mind that domain vectors are stored as rows, not columns.
    
    **ARGUMENTS**
    rvecs
        A (3,3) array representing a domain matrix.
    
    **RETURNS**
    new_rvecs
        A (3,3) array representing a lower-diagonal form of rvecs.
    rot
        A (3,3) array representing the rotation matrix to go from rvecs to new_rvecs.
    """
    assert rvecs.shape == (3,3), "Only 3D periodic systems supported!"
    new_rvecs = np.zeros(rvecs.shape)
    A = rvecs[0]
    B = rvecs[1]
    C = rvecs[2]
    assert np.dot(np.cross(A,B),C) > 0, "Domain vectors should form right-handed basis!"
    # Vector a.
    new_rvecs[0,0] = np.linalg.norm(A) # a vector
    # Vector b.
    new_rvecs[1,0] = np.dot(B,A)/new_rvecs[0,0]
    new_rvecs[1,1] = np.linalg.norm(np.cross(A,B))/new_rvecs[0,0]
    # Vector c.
    new_rvecs[2,0] = np.dot(C,A)/new_rvecs[0,0]
    new_rvecs[2,1] = (np.dot(B,C) - new_rvecs[1,0]*new_rvecs[2,0])/new_rvecs[1,1]
    new_rvecs[2,2] = np.sqrt( np.dot(C,C) - new_rvecs[2,0]**2 - new_rvecs[2,1]**2 )
    # Transformation matrix.
    rot = np.zeros(rvecs.shape)
    rot[0] = np.cross(B,C)
    rot[1] = np.cross(C,A)
    rot[2] = np.cross(A,B)
    rot = np.dot(new_rvecs.T,rot)/np.abs(np.linalg.det(rvecs))
    return new_rvecs, rot


def get_random_vel_press(mass, temp):
    """
    Generates symmetric tensor of barostat velocities.
    
    **ARGUMENTS**
    mass
        The Barostat mass.
    temp
        The temperature at which the velocities are selected.
    """
    shape = (3, 3)
    # Generate random 3x3 tensor.
    rand = np.random.normal(0, np.sqrt(mass*boltzmann*temp), shape)/mass
    vel_press = np.zeros(shape)
    # Create initial symmetric pressure velocity tensor.
    for i in range(3):
        for j in range(3):
            if i >= j:
                vel_press[i,j] = rand[i,j]
            else:
                vel_press[i,j] = rand[j,i]
            # Correct for p_ab = p_ba, hence only 1 dof if a != b.
            if i != j:
                vel_press[i,j] /= np.sqrt(2)
    return vel_press


def get_ndof_baro(dim, anisotropic, vol_constraint):
    """    
    Calculates the number of degrees of freedom associated with the simulation domain fluctuation.
    
    **ARGUMENTS**
    dim
        The dimensionality of the system.
    anisotropic
        Boolean value determining whether anisotropic domain fluctuations are allowed.
    vol_constraint
        Boolean value determining whether the domain volume can change.
    """
    ndof = 1
    # Degrees of freedom for a symmetric domain tensor.
    if anisotropic:
        ndof = dim*(dim + 1)//2
    # Decrease the number of dof by one if volume is constant.
    if vol_constraint:
        ndof -= 1
    # Verify at least one degree of freedom is left.
    if ndof == 0:
        raise AssertionError("Isotropic barostat called with a volume constraint.")
    return ndof


def stabilized_cholesky_decomp(mat):
    """
    Do LDL^T and transform to MM^T with negative diagonal entries of D put equal to zero.
    Assume mat is square and symmetric (but not necessarily positive definite).
    """
    if np.all(np.linalg.eigvals(mat) > 0):
        return np.linalg.cholesky(mat)  # usual cholesky decomposition
    else:
        n = int(mat.shape[0])
        D = np.zeros(n)
        L = np.zeros(mat.shape)
        for i in np.arange(n):
            L[i, i] = 1
            for j in np.arange(i):
                L[i, j] = mat[i, j]
                for k in np.arange(j):
                    L[i, j] -= L[i, k] * L[j, k] * D[k]
                if abs(D[j]) > 1e-12:
                    L[i, j] *= 1.0/D[j]
                else:
                    L[i, j] = 0
            D[i] = mat[i, i]
            for k in np.arange(i):
                D[i] -= L[i, k] * L[i, k] * D[k]
        D = np.sqrt(D.clip(min=0))
        return L*D




