#!/usr/bin/env python

from __future__ import print_function

import os


__all__ = ['write_if_changed']


def write_if_changed(fn, s_new):
    if os.path.isfile(fn):
        # read the entire file
        with open(fn) as f:
            s_old = f.read()
        if s_new == s_old:
            print('File %s needs no update. Skipping.' % fn)
            return

    # write the new file to dis
    print('Writing new or updated %s' % fn)
    with open(fn, 'w') as f:
        f.write(s_new)
