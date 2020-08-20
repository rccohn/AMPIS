# Copyright (c) 2020 Ryan Cohn and Elizabeth Holm. All rights reserved.
# Licensed under the MIT License (see LICENSE for details)
# Written by Ryan Cohn
"""
Base module for AMPIS. All functions are contained in 
analyze, data_utils, structures, visualize, and applications.
"""
from pathlib import Path

from . import analyze
from . import data_utils
from . import structures
from . import visualize
from . import applications

__all__ = ['analyze', 'data_utils', 'structures', 'visualize', 'applications']
with open(Path(Path(__file__).parent,'..','setup.py'), 'rb') as f:
    setup_lines = f.readlines()
for line in setup_lines:
    line = line.decode('utf-8')
    if 'version' in line:
        verline = line
        for char in ([' ', 'version', '=', ',', '\n',"'",'"']):
            verline = verline.replace(char, '')
        __version__ = verline
        break

__all__.append('__version__')
del verline, char, line, setup_lines

