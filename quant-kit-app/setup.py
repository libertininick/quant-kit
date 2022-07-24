#!/usr/bin/env python

import re
import setuptools
from distutils.util import convert_path

MODULE_NAME = "quant_kit_app"
with open(convert_path(f"src/{MODULE_NAME}/__init__.py")) as fp:
    LOCKSTEP_VERSION = re.search(
        pattern=r'\s*__version__\s*=\s*"(.+)"', 
        string=fp.read(),
        flags=re.MULTILINE
    ).group(1)

if __name__ in ["__main__", "builtins"]:
    setuptools.setup(
        install_requires=[
            f"quant-kit-core=={LOCKSTEP_VERSION}",
        ]
    )