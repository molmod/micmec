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


"""Screen logger. 

The ``log`` object is an instance of the ``molmod.log.ScreenLog`` class.
The logger comes with a timer infrastructure, which is also implemented in the ``molmod.log`` module.
"""

import atexit

from molmod.log import ScreenLog, TimerGroup


__all__ = ["log", "timer"]

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


