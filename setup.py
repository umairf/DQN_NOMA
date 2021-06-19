# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:18:37 2020

@author: Umair
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("DQN.pyx" )

)

setup(
    ext_modules = cythonize("Buffer.pyx")

)
setup(
    ext_modules = cythonize("agentcode.pyx")

)

setup(
    ext_modules = cythonize("model.pyx")

)

setup(
    ext_modules = cythonize("execute1.pyx")

)

