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

low = 998290676 #Initial densities in kg/m^3 
high = 99829067611
count = 0
for density in tqdm.tqdm(range( low , high ,5*10**8)):
    tov = TOV( density , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, True, log_active)
    tov.ComputeTOV()    
    tov.hbar_into_txt(count)
    tov.radius_into_txt(count)
    tov.density_into_txt()
    count += 1
tov.Plot_all_hbar(count)

