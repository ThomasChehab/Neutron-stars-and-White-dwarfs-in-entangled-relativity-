from TOV import *

import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.constants as cst
import numpy as np

import os

#################################################################
#Changement :
c2 = cst.c**2


#################################################################


PhiInit = 1
PsiInit = 0
option = 1
radiusMax_in = 50000
radiusMax_out = 100000000
Npoint = 50000
log_active = True
dilaton = True

retro = False
initDensity = 1000 *cst.eV*10**6/(cst.c**2*cst.fermi**3)
tov = TOV(initDensity , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton, log_active,retro)
PhiInit = tov.find_dilaton_center()[0]
tov = TOV(initDensity , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton, log_active,retro)
tov.ComputeTOV()
tov.hbar_into_txt()
tov.radius_into_txt()

retro = True
tov = TOV(initDensity , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton, log_active,retro)
PhiInit = tov.find_dilaton_center()[0]
tov = TOV(initDensity , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton, log_active,retro)
tov.ComputeTOV()
tov.hbar_retro_into_txt()
tov.radius_retro_into_txt()
tov.Plot()

