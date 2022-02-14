#!/usr/bin/env python
# File name: test_chk.py
# Description: testing
# Author: Joachim Vandewalle
# Date: 26-10-2021

from molmod.io.chk import *

test = {"dictionary": {"key": 1}}

dump_chk("test.chk", test)
