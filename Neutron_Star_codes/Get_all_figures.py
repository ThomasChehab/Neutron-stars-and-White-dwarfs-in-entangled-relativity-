from TOV import *
import matplotlib 
import matplotlib.pyplot as plt
import scipy.constants as cst
import numpy as np
from tqdm import tqdm
from lal import  C_SI, HBAR_SI, H_SI, G_SI, MSUN_SI
M_sun = MSUN_SI
import pickle
import os

c2 = cst.c**2
PhiInit = 1
PsiInit = 0
option = 1
log_active = False
density = 1000
initDensity = density*cst.eV*10**6/(cst.c**2*cst.fermi**3)
dilaton = True
radiusStep = 10
initRadius = 0.000001
precision = 1e-8

retro = False
tov = TOV(initRadius, initDensity, PsiInit, PhiInit, radiusStep, option, dilaton, precision, retro)
PhiInit = tov.find_dilaton_center()[0]
tov.Compute()
tov.hbar_into_txt() 
tov.radius_into_txt() 

PhiInit = 1
retro = True
tov = TOV(initRadius, initDensity, PsiInit, PhiInit, radiusStep, option, dilaton, precision, retro)
PhiInit = tov.find_dilaton_center()[0]
tov = TOV(initRadius, initDensity, PsiInit, PhiInit, radiusStep, option, dilaton, precision, retro)
tov.Compute()
tov.hbar_retro_into_txt() 
tov.radius_retro_into_txt()
tov.Plot()



