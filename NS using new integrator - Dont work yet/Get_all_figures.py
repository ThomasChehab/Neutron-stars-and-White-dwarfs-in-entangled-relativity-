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
radiusInit = 0.00001
radiusMax_in = 20000000
radiusMax_out = 100000000
Npoint = 50000
#log_active = False
log_active = False
density = 1000
dilaton_active = True
precision = 1e-2
retro = True


#tov = TOV(density, PsiInit, PhiInit, radiusInit, radiusMax_in,  radiusMax_out, Npoint, option, dilaton_active, log_active, precision, retro)
#PhiInit = tov.find_dilaton_center()[0]
#tov = TOV(density, PsiInit, PhiInit, radiusInit, radiusMax_in,  radiusMax_out, Npoint, option, dilaton_active, log_active, precision, retro)
#tov.ComputeTOV()

retro = False
tov = TOV(density, PsiInit, PhiInit, radiusInit, radiusMax_in,  radiusMax_out, Npoint, option, dilaton_active, log_active, precision, retro)
PhiInit = tov.find_dilaton_center()[0]
tov = TOV(density, PsiInit, PhiInit, radiusInit, radiusMax_in,  radiusMax_out, Npoint, option, dilaton_active, log_active, precision, retro)
tov.Compute()
tov.hbar_into_txt() 
tov.radius_into_txt() 



PhiInit = 1
retro = True
PhiInit = tov.find_dilaton_center()[0]
tov = TOV(density, PsiInit, PhiInit, radiusInit, radiusMax_in,  radiusMax_out, Npoint, option, dilaton_active, log_active, precision, retro)
tov.Compute()
tov.hbar_retro_into_txt()
tov.radius_retro_into_txt()
tov.Plot()


