#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys

def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    run([sys.executable, "calibrate_cf.py"])
    run([sys.executable, "calibrate_lc.py"])
    run([sys.executable, "stability.py"])
    run([sys.executable, "ctm.py"])

if __name__ == "__main__":
    main()
