#!/usr/bin/env python
# File name: log.py
# Description: Screenlogger for the micromechanical model.
# Author: Joachim Vandewalle
# Date: 18-11-2021

"""
This module holds the main screen loging object of MicMec. 
The ``log`` object is an instance off the ``ScreenLog`` class in the module ``molmod.log``.
The logger also comes with a timer infrastructure, which is also implemented in the ``molmod.log`` module.
"""

import atexit

from molmod.log import ScreenLog, TimerGroup


__all__ = ['log', 'timer']


head_banner = r"""
Welcome to MicMec.
 ||==================================||  
 ||   __      __    ___     _____    ||
 ||    \      /      |     /    /    ||
 ||    |\    /|      |     |         ||
 ||    | \  / |      |     |         ||
 ||   _|_ \/ _|_    _|_    |____|    ||
 ||   __      __  _______   _____    ||
 ||    \      /    |    /  /    /    ||
 ||    |\    /|    |____   |         ||
 ||    | \  / |    |       |         ||
 ||   _|_ \/ _|_  _|____|  |____|    ||
 ||                                  ||
 ||==================================||
"""


foot_banner = r"""
End of file. Come back soon!
"""

timer = TimerGroup()
log = ScreenLog("MICMEC", "v1.0", head_banner, foot_banner, timer)
atexit.register(log.print_footer)
