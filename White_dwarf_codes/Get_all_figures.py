from TOV import *
import matplotlib
import matplotlib.pyplot as plt
import scipy.constants as cst
import numpy as np
import os
import tqdm 

c2 = cst.c**2
PhiInit = 1
PsiInit = 0
option = 1
radiusMax_in = 20000000
radiusMax_out = 100000000
Npoint = 50000
log_active = False # change for True for seeing star's data
lowest_density = 1e9 # kg/m-3
highest_density = 1e13
densities = np.linspace(np.log(lowest_density), np.log(highest_density), 249)
densities = np.exp(densities)
count = 0
for density in (densities):
    tov = TOV( density , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, True, log_active)
    tov.ComputeTOV()    
    tov.hbar_into_txt(count)
    tov.radius_into_txt(count)
    tov.density_into_txt()
    count += 1
tov.Plot_all_hbar()

