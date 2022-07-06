#!/usr/bin/env python
# File name: log.py
# Description: Screenlogger for the micromechanical model.
# Author: Joachim Vandewalle
# Date: 18-11-2021

"""The main screen loging object of MicMec. 

The `log` object is an instance off the `ScreenLog` class in the module `molmod.log`.
The logger also comes with a timer infrastructure, which is also implemented in the `molmod.log` module.
"""

import atexit

from molmod.log import ScreenLog, TimerGroup


__all__ = ['log', 'timer']

#================================================================================
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
head_banner = r"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                     _____    _____    _____      _______     
                      |  \    /  |      | |      / /  /_/
                      | \ \  / / |      | |     / /
                      | |\ \/ /| |      | |     | |    
                      | | \__/ | |      | |     | |   ___
                     _|_|_    _|_|_    _|_|_    |_|___|_|

                     _____    _____  _________    _______
                      |  \    /  |    | |  |_|   / /  /_/
                      | \ \  / / |    | | __    / /
                      | |\ \/ /| |    | |/_/    | |    
                      | | \__/ | |    | |  ___  | |   ___
                     _|_|_    _|_|_  _|_|__|_|  |_|___|_|

                                 ________________
                                /__/__/__/__/__/|
                               /__/__/__/__/__/||
                              /__/__/__/__/__/|||
                              |__|__|__|__|__||||
                              |__|__|__|__|__||||
                              |__|__|__|__|__|||/
                              |__|__|__|__|__||/
                              |__|__|__|__|__|/


                             Welcome to MicMec 1.0,
                         written by Joachim Vandewalle
                        (joachim.vandewalle@hotmail.be).  

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                
"""


foot_banner = r"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                          End of file. Come back soon!

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

timer = TimerGroup()
log = ScreenLog("MICMEC", "1.0", head_banner, foot_banner, timer)
atexit.register(log.print_footer)
