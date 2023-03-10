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


"""Basic trajectory analysis routines."""

import numpy as np

from molmod import boltzmann, pascal

from micmec.log import log
from micmec.analysis.utils import get_slice, get_time


__all__ = [
    "plot_energies",
    "plot_temperature",
    "plot_pressure",
    "plot_temp_dist",
    "plot_press_dist",
    "plot_volume_dist",
    "plot_domain_pars",
]


def plot_energies(f, fn_png="energies.png", **kwargs):
    """Make a plot of the potential, kinetic, total and conserved energy as a function of time.

    Parameters
    ----------
    f : h5py.File
        An HDF5 file containing the trajectory data.
    fn_png : str, optional
        The PNG filename to write the figure to.

    Notes
    -----
    The optional arguments of the ``get_slice`` function are also accepted in the form of keyword arguments.
    The units for making the plot are taken from the screen logger.
    This type of plot is essential for checking the sanity of a simulation.
    """
    import matplotlib.pyplot as pt

    start, end, step = get_slice(f, **kwargs)

    epot = f["trajectory/epot"][start:end:step] / log.energy.conversion
    ekin = f["trajectory/ekin"][start:end:step] / log.energy.conversion
    time, tlabel = get_time(f, start, end, step)

    pt.clf()
    pt.plot(time, epot, "k-", label="potential")
    pt.plot(time, ekin, "b-", label="kinetic")
    if "trajectory/etot" in f:
        etot = f["trajectory/etot"][start:end:step] / log.energy.conversion
        pt.plot(time, etot, "r-", label="total")
    if "trajectory/econs" in f:
        econs = f["trajectory/econs"][start:end:step] / log.energy.conversion
        pt.plot(time, econs, "g-", label="conserved")
    pt.xlim(time[0], time[-1])
    pt.xlabel(tlabel)
    pt.ylabel("ENERGY [%s]" % log.energy.notation)
    pt.legend(loc=0)
    pt.grid()
    pt.savefig(fn_png)


def plot_temperature(f, fn_png="temperature.png", **kwargs):
    """Make a plot of the temperature as a function of time.

    Parameters
    ----------
    f : h5py.File
        An HDF5 file containing the trajectory data.
    fn_png : str, optional
        The PNG filename to write the figure to.

    Notes
    -----
    The optional arguments of the ``get_slice`` function are also accepted in the form of keyword arguments.
    The units for making the plot are taken from the screen logger.
    This type of plot is essential for checking the sanity of a simulation.
    """
    import matplotlib.pyplot as pt

    start, end, step = get_slice(f, **kwargs)
    temp = f["trajectory/temp"][start:end:step]
    time, tlabel = get_time(f, start, end, step)
    pt.clf()
    pt.plot(time, temp, "k-")
    pt.xlim(time[0], time[-1])
    pt.xlabel(tlabel)
    pt.ylabel("TEMPERATURE [K]")
    pt.grid()
    pt.savefig(fn_png)


def plot_pressure(f, fn_png="pressure.png", window=1, **kwargs):
    """Make a plot of the pressure as a function of time.

    Parameters
    ----------
    f : h5py.File
        An HDF5 file containing the trajectory data.
    fn_png : str, optional
        The PNG filename to write the figure to.
    window : int, optional
        The window over which the pressure is averaged.

    Notes
    -----
    The optional arguments of the ``get_slice`` function are also accepted in the form of keyword arguments.
    The units for making the plot are taken from the screen logger.
    This type of plot is essential for checking the sanity of a simulation.
    """
    import matplotlib.pyplot as pt

    start, end, step = get_slice(f, **kwargs)

    press = f["trajectory/press"][start:end:step]
    time, tlabel = get_time(f, start, end, step)

    press_av = np.zeros(len(press) + 1 - window)
    time_av = np.zeros(len(press) + 1 - window)
    for i in range(len(press_av)):
        press_av[i] = press[i:(i + window)].sum() / window
        time_av[i] = time[i]
    pt.clf()
    pt.plot(
        time_av,
        press_av / (1e9 * pascal),
        "k-",
        label="simulation (%.3f MPa)" % (press.mean() / (1e6 * pascal)),
    )
    pt.xlim(time[0], time[-1])
    pt.xlabel(tlabel)
    pt.ylabel("PRESSURE [GPa]")
    pt.legend(loc=0)
    pt.grid
    pt.savefig(fn_png)


def plot_temp_dist(
    f, fn_png="temp_dist.png", temp=None, ndof=None, select=None, **kwargs
):
    """Plot the distribution of the weighted nodal velocities (temperature).

    Parameters
    ----------
    f : h5py.File
        An HDF5 file containing the trajectory data.
    fn_png : str, optional
        The PNG filename to write the figure to.
    temp : float
        The (expected) average temperature.
    select : array_like, optional
        A list of node indices that should be considered for the analysis.
        By default, information from all nodes is combined.
    start, end, step, max_sample : int, optional
       The optional arguments of the ``get_slice`` function are also accepted in the form of keyword arguments.

    Notes
    -----
    This type of plot is essential for checking the sanity of a simulation.
    """
    import matplotlib.pyplot as pt
    from matplotlib.ticker import MaxNLocator

    start, end, step = get_slice(f, **kwargs)

    # Make an array with the weights used to compute the temperature.
    if select is None:
        weights = np.array(f["system/masses"]) / boltzmann
    else:
        weights = np.array(f["system/masses"])[select] / boltzmann

    if select is None:
        # Just load the temperatures from the output file.
        temps = f["trajectory/temp"][start:end:step]
    else:
        # Compute the temperatures of the subsystem.
        temps = []
        for i in range(start, end, step):
            temp = ((f["trajectory/vel"][i, select] ** 2).mean(axis=1) * weights).mean()
            temps.append(temp)
        temps = np.array(temps)

    if temp is None:
        temp = temps.mean()

    # A) SYSTEM
    if select is None:
        nnodes = f["system/pos"].shape[0]
    else:
        nnodes = 3 * len(select)
    if ndof is None:
        ndof = f["trajectory"].attrs.get("ndof")
    if ndof is None:
        ndof = 3 * nnodes
    sigma = temp * np.sqrt(2.0 / ndof)
    temp_step = sigma / 5

    # Setup the temperature grid and make the histogram.
    temp_grid = np.arange(max(0, temp - 3 * sigma), temp + 5 * sigma, temp_step)
    counts = np.histogram(temps.ravel(), bins=temp_grid)[0]
    total = float(len(temps))

    # Transform into empirical pdf and cdf.
    emp_sys_pdf = counts / total
    emp_sys_cdf = counts.cumsum() / total

    # B) Make the plots
    pt.clf()
    ax1 = pt.subplot(2, 1, 1)
    # pt.title("System (ndof=%i)" % ndof)
    scale = 1 / emp_sys_pdf.max()
    pt.plot(
        emp_sys_pdf * scale,
        "k-",
        drawstyle="steps-pre",
        label="simulation (%.0f K)" % (temps.mean()),
    )
    pt.axvline(temp, color="r", ls="--")
    pt.axvline(temps.mean(), color="k", ls="--")
    pt.ylim(ymin=0)
    pt.ylabel("RESCALED PDF")
    pt.legend(loc=0)
    ax1.grid()
    pt.gca().get_xaxis().set_major_locator(MaxNLocator(nbins=5))

    ax2 = pt.subplot(2, 1, 2)
    pt.plot(emp_sys_cdf, "k-", drawstyle="steps-pre")
    pt.axvline(temp, color="r", ls="--")
    pt.axvline(temps.mean(), color="k", ls="--")
    pt.ylim(0, 1)
    pt.gca().get_xaxis().set_major_locator(MaxNLocator(nbins=5))
    pt.ylabel("CDF")
    ax2.grid()
    pt.xlabel("TEMPERATURE [%s]" % log.temperature.notation)
    pt.savefig(fn_png, dpi=500.0)


def plot_press_dist(
    f, temp, fn_png="press_dist.png", press=None, ndof=None, select=None, **kwargs
):
    """Plot the distribution of the internal pressure.

    Parameters
    ----------
    f : h5py.File
        An HDF5 file containing the trajectory data.
    fn_png : str, optional
        The PNG filename to write the figure to.
    temp : float
        The (expected) average temperature.
    press : float, optional
        The (expected) average pressure.
    select : array_like, optional
        A list of node indices that should be considered for the analysis.
        By default, information from all nodes is combined.
    start, end, step, max_sample : int, optional
       The optional arguments of the ``get_slice`` function are also accepted in the form of keyword arguments.

    Notes
    -----
    This type of plot is essential for checking the sanity of a simulation.
    """
    import matplotlib.pyplot as pt
    from matplotlib.ticker import MaxNLocator

    start, end, step = get_slice(f, **kwargs)

    # Make an array with the weights used to compute the temperature.
    if select is None:
        weights = np.array(f["system/masses"]) / boltzmann
    else:
        weights = np.array(f["system/masses"])[select] / boltzmann

    if select is None:
        # Just load the temperatures from the output file.
        temps = f["trajectory/temp"][start:end:step]
    else:
        # Compute the temperatures of the subsystem.
        temps = []
        for i in range(start, end, step):
            temp = ((f["trajectory/vel"][i, select] ** 2).mean(axis=1) * weights).mean()
            temps.append(temp)
        temps = np.array(temps)

    if temp is None:
        temp = temps.mean()

    presss = f["trajectory/press"][start:end:step]
    if press is None:
        press = presss.mean()

    # A) SYSTEM
    sigma = np.std(presss)
    press_step = sigma / 5

    # Setup the pressure grid and make the histogram.
    press_grid = np.arange(press - 5 * sigma, press + 5 * sigma, press_step)
    counts = np.histogram(presss.ravel(), bins=press_grid)[0]
    total = float(len(presss))

    # Transform into empirical pdf and cdf.
    emp_sys_pdf = counts / total
    emp_sys_cdf = counts.cumsum() / total

    # B) Make the plots.
    pt.clf()
    ax1 = pt.subplot(2, 1, 1)
    scale = 1 / emp_sys_pdf.max()
    pt.plot(
        emp_sys_pdf * scale,
        "k-",
        drawstyle="steps-pre",
        label="simulation (%.3f MPa)" % (presss.mean() / (1e6 * pascal)),
    )
    pt.axvline(press, color="r", ls="--")
    pt.axvline(presss.mean(), color="k", ls="--")
    pt.ylim(ymin=0)
    pt.ylabel("RESCALED PDF")
    ax1.grid()
    pt.legend(loc=0)
    pt.gca().get_xaxis().set_major_locator(MaxNLocator(nbins=5))

    ax2 = pt.subplot(2, 1, 2)
    pt.plot(emp_sys_cdf, "k-", drawstyle="steps-pre")
    pt.axvline(press, color="r", ls="--")
    pt.axvline(presss.mean(), color="k", ls="--")
    pt.ylim(0, 1)
    ax2.grid()
    pt.gca().get_xaxis().set_major_locator(MaxNLocator(nbins=5))
    pt.ylabel("CDF")
    pt.xlabel("PRESSURE [MPa]")
    pt.savefig(fn_png, dpi=500.0)


def plot_volume_dist(f, fn_png="volume_dist.png", temp=None, press=None, **kwargs):
    """Plot the distribution of the volume.

    Parameters
    ----------
    f : h5py.File
        An HDF5 file containing the trajectory data.
    fn_png : str, optional
        The PNG filename to write the figure to
    temp : float, optional
        The (expected) average temperature.
    press : float, optional
        The (expected) average pressure.
    start, end, step, max_sample : int, optional
       The optional arguments of the ``get_slice`` function are also accepted in the form of keyword arguments.

    Notes
    -----
    This type of plot is essential for checking the sanity of a simulation.
    """
    import matplotlib.pyplot as pt
    from matplotlib.ticker import MaxNLocator

    start, end, step = get_slice(f, **kwargs)

    if temp is None:
        # Make an array of the temperature.
        temps = f["trajectory/temp"][start:end:step]
        temp = temps.mean()

    if press is None:
        # Make an array of the pressure.
        presss = f["trajectory/press"][start:end:step]
        press = presss.mean()

    # Make an array of the domain volume.
    vols = f["trajectory/volume"][start:end:step]
    vol0 = vols.mean()

    sigma = np.std(vols)
    vol_step = sigma / 5

    # Setup the volume grid and make the histogram.
    vol_grid = np.arange(vol0 - 3 * sigma, vol0 + 3 * sigma, vol_step)
    counts = np.histogram(vols.ravel(), bins=vol_grid)[0]
    total = float(len(vols))

    # Transform into empirical pdf and cdf.
    emp_sys_pdf = counts / total
    emp_sys_cdf = counts.cumsum() / total

    # Make the plots.
    pt.clf()
    ax1 = pt.subplot(2, 1, 1)
    scale = 1 / emp_sys_pdf.max()
    pt.plot(emp_sys_pdf * scale, "k-", drawstyle="steps-pre")
    pt.ylim(ymin=0)
    pt.ylabel("RESCALED PDF")
    ax1.grid()
    pt.gca().get_xaxis().set_major_locator(MaxNLocator(nbins=5))

    ax2 = pt.subplot(2, 1, 2)
    pt.plot(emp_sys_cdf, "k-", drawstyle="steps-pre")
    pt.ylim(0, 1)
    pt.gca().get_xaxis().set_major_locator(MaxNLocator(nbins=5))
    pt.ylabel("CDF")
    pt.xlabel("VOLUME [A^3]")
    ax2.grid()

    pt.savefig(fn_png)


def plot_domain_pars(f, fn_png="domain_pars.png", **kwargs):
    """Make a plot of the domain parameters as a function of time.

    Parameters
    ----------
    f : h5py.File
        An HDF5 file containing the trajectory data.
    fn_png : str, optional
        The PNG filename to write the figure to.

    Notes
    -----
    The optional arguments of the ``get_slice`` function are also accepted in the form of keyword arguments.
    The units for making the plot are taken from the screen logger.
    This type of plot is essential for checking the sanity of a simulation.
    """
    import matplotlib.pyplot as pt

    start, end, step = get_slice(f, **kwargs)

    if "trajectory/domain" in f:
        domain = f["trajectory/domain"][start:end:step] / log.length.conversion
    else:
        if "trajectory/cell" in f:
            domain = f["trajectory/cell"][start:end:step] / log.length.conversion
        else:
            raise IOError("File does not contain a domain trajectory.")
    lengths = np.sqrt((domain**2).sum(axis=2))
    time, tlabel = get_time(f, start, end, step)
    nvec = lengths.shape[1]

    def get_angle(i0, i1):
        return (
            np.arccos(
                np.clip(
                    (domain[:, i0] * domain[:, i1]).sum(axis=1)
                    / lengths[:, i0]
                    / lengths[:, i1],
                    -1,
                    1,
                )
            )
            / log.angle.conversion
        )

    pt.clf()
    if nvec == 3:
        ax1 = pt.subplot(2, 1, 1)
        pt.plot(time, lengths[:, 0], "r-", label="a")
        pt.plot(time, lengths[:, 1], "g-", label="b")
        pt.plot(time, lengths[:, 2], "b-", label="c")
        pt.xlim(time[0], time[-1])
        pt.ylabel("LENGTHS [%s]" % log.length.notation)
        ax1.grid()
        pt.legend(loc=0)

        alpha = get_angle(1, 2)
        beta = get_angle(2, 0)
        gamma = get_angle(0, 1)
        ax2 = pt.subplot(2, 1, 2)
        pt.plot(time, alpha, "r-", label="alpha")
        pt.plot(time, beta, "g-", label="beta")
        pt.plot(time, gamma, "b-", label="gamma")
        pt.xlim(time[0], time[-1])
        pt.xlabel(tlabel)
        pt.ylabel("ANGLES [%s]" % log.angle.notation)
        ax2.grid()
        pt.legend(loc=0)
    elif nvec == 2:
        ax1 = pt.subplot(2, 1, 1)
        pt.plot(time, lengths[:, 0], "r-", label="a")
        pt.plot(time, lengths[:, 1], "g-", label="b")
        pt.xlim(time[0], time[-1])
        pt.ylabel("LENGTHS [%s]" % log.length.notation)
        ax1.grid()
        pt.legend(loc=0)

        gamma = get_angle(0, 1)
        ax2 = pt.subplot(2, 1, 2)
        pt.plot(time, gamma, "b-", label="gamma")
        pt.xlim(time[0], time[-1])
        pt.xlabel(tlabel)
        pt.ylabel("ANGLE [%s]" % log.angle.notation)
        ax2.grid()
        pt.legend(loc=0)
    elif nvec == 1:
        pt.plot(time, lengths[:, 0], "k-")
        pt.xlim(time[0], time[-1])
        pt.xlabel(tlabel)
        pt.ylabel("LENGTHS [%s]" % log.length.notation)
        pt.grid()
    else:
        raise ValueError("Cannot plot domain parameters if the system is not periodic.")

    pt.savefig(fn_png)
