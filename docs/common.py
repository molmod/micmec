#!/usr/bin/env python
import os


__all__ = ["write_if_changed"]


def write_if_changed(fn, s_new):
    if os.path.isfile(fn):
        with open(fn) as f:
            s_old = f.read()
        if s_new == s_old:
            print("File %s needs no update. Skipping." % fn)
            return

    print("Writing new or updated %s" % fn)
    with open(fn, "w") as f:
        f.write(s_new)
