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
log_active = False # Change for True for seeing star's data
dilaton_active = True # Change for false for deactivating scalar field
lowest_density = 1e9 # kg/m3
highest_density = 1e13
densities = np.linspace(np.log(lowest_density), np.log(highest_density), 5)
densities = np.exp(densities)
count = 0

folder_list = ['./hbar_folder', './init_density_folder', './radius_folder', './star_hbar_folder', './star_radius_folder']

if verify_folder(folder_list):
    Plot_all_hbar() #plot the variation of hbar versus the distance for various densities
else:
    print('All folders are not present. Let\'s compute the data')
    verify()
    for density in tqdm.tqdm(densities):
        tov = TOV(density , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton_active, log_active,count)
        PhiInit = tov.find_dilaton_center()[0] #find the value of phi at the center to have phi = 1 at infinity
        tov = TOV(density , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton_active, log_active,count)
        tov.ComputeTOV() #computing star's data

        tov.recover_star_radius() # saving star's data
        tov.recover_hbar_star()
        tov.hbar_into_txt(count)
        tov.radius_into_txt(count)
        tov.density_into_txt()
        count += 1
    Plot_all_hbar()
