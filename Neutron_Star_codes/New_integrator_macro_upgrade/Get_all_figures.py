#from TOV_hbar import *
from TOV import *
import matplotlib
import matplotlib.pyplot as plt
import scipy.constants as cst
import scipy.optimize
import numpy as np
import math
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, splrep
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.transforms import Bbox
from tqdm import tqdm



def verify():
    if os.path.exists('./NS_data.dat'):
        os.remove('./NS_data.dat')
    if os.path.exists('./NS_hbar_dependency_data.dat'):
        os.remove('./NS_hbar_dependency_data.dat')
verify()

def run_GR(rho_cen):
    dependence = 2 #hbar^2
    PhiInit = 1
    PsiInit = 0
    option = 1
    radiusMax_in = 40000#50000
    radiusMax_out = 100000000
    Npoint = 50000
    retro = False

    log_active = False
    dilaton_active = False
    rhoInit = rho_cen*cst.eV*10**6/(cst.c**2*cst.fermi**3)
    tov = TOV(rhoInit, PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton_active, log_active,retro, dependence)
    tov.ComputeTOV_normalization()

    radiusStar = tov.radiusStar
    mass_ADM = tov.massADM / (1.989*10**30) # in solar mass
    phi_star = tov.phiStar
    phi_0 = tov.Phi[0]
    SoS_c_max = np.max(tov.v_c)

    return radiusStar, mass_ADM, phi_star, phi_0, SoS_c_max


def run_ER(rho_cen):
    dependence = 2
    PhiInit = 1
    PsiInit = 0
    option = 1
    radiusMax_in = 40000#50000
    radiusMax_out = 100000000
    Npoint = 50000
    retro = False
    log_active = False
    dilaton_active = True
    rhoInit = rho_cen*cst.eV*10**6/(cst.c**2*cst.fermi**3)
    tov = TOV(rhoInit, PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton_active, log_active, retro, dependence)
    tov.ComputeTOV_normalization()

    radiusStar = tov.radiusStar
    mass_ADM = tov.massADM / (1.989*10**30) # in solar mass
    phi_star = tov.phiStar
    phi_0 = tov.Phi[0]
    SoS_c_max = np.max(tov.v_c)

    return radiusStar, mass_ADM, phi_star, phi_0, SoS_c_max


def run_ER_retro(rho_cen):
    dependence = 2
    PhiInit = 1
    PsiInit = 0
    option = 1
    retro = True
    radiusMax_in = 40000#50000
    radiusMax_out = 100000000
    Npoint = 50000
    log_active = False
    dilaton_active = True
    rhoInit = rho_cen*cst.eV*10**6/(cst.c**2*cst.fermi**3)
    tov = TOV(rhoInit , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton_active, log_active,retro, dependence)
    PhiInit = tov.find_dilaton_center()[0]
    tov = TOV(rhoInit , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton_active, log_active,retro, dependence)
    tov.ComputeTOV()
    radiusStar = tov.radiusStar
    mass_ADM = tov.massADM / (1.989*10**30) # in solar mass
    phi_star = tov.phiStar
    phi_0 = tov.Phi[0]
    SoS_c_max = np.max(tov.v_c)

    return radiusStar, mass_ADM, phi_star, phi_0, SoS_c_max


n = 1000 # 4000
den_space = np.linspace(100,2000,num=n)
size_a = np.array([])
mass_a = np.array([])
delta_hbar_a = np.array([])
delta_hbar0_a = np.array([])
vsurc_a = np.array([])

size_a_GR = np.array([])
mass_a_GR = np.array([])
delta_hbar_a_GR = np.array([])
vsurc_a_GR = np.array([])

size_a_ER_retro = np.array([])
mass_a_ER_retro = np.array([])
delta_hbar_a_ER_retro = np.array([])
delta_hbar0_a_ER_retro = np.array([])
vsurc_a_ER_retro = np.array([])

retro = False


#
# for den in tqdm(den_space):
#
#
#     size_e3_GR, mass_GR, phi_s_GR, phi_0_GR, vsurc_GR = run_GR(den) # Run in general relativity
#
#     size_a_GR = np.append(size_a_GR,size_e3_GR/1e3)
#     mass_a_GR = np.append(mass_a_GR, mass_GR)
#     delta_hbar_a_GR = np.append(delta_hbar_a_GR, - (phi_s_GR-1) / 2.)
#     vsurc_a_GR = np.append(vsurc_a_GR,vsurc_GR)
#
#
#     size_e3, mass, phi_s, phi_0, vsurc = run_ER(den) # Run in entangled relativity
#
#     size_a = np.append(size_a,size_e3/1e3)
#     mass_a = np.append(mass_a, mass)
#     delta_hbar_a = np.append(delta_hbar_a, - (phi_s-1) / 2.)
#     delta_hbar0_a = np.append(delta_hbar0_a, - (phi_0-1) / 2.)
#     vsurc_a = np.append(vsurc_a,vsurc)
#
#
#     size_e3_ER_retro, mass_ER_retro, phi_s_ER_retro, phi_0_ER_retro, vsurc_ER_retro = run_ER_retro(den) # Run in entangled relativity with retroation
#
#     size_a_ER_retro = np.append(size_a_ER_retro,size_e3_ER_retro/1e3)
#     mass_a_ER_retro = np.append(mass_a_ER_retro, mass_ER_retro)
#     delta_hbar_a_ER_retro = np.append(delta_hbar_a_ER_retro, - (phi_s_ER_retro-1) / 2.)
#     delta_hbar0_a_ER_retro = np.append(delta_hbar0_a_ER_retro, - (phi_0_ER_retro-1) / 2.)
#     vsurc_a_ER_retro = np.append(vsurc_a_ER_retro,vsurc_ER_retro)




all_a = [size_a,mass_a,delta_hbar_a,delta_hbar0_a,vsurc_a]
all_a_GR = [size_a_GR,mass_a_GR,delta_hbar_a_GR,vsurc_a_GR]
all_a_ER_retro = [size_a_ER_retro,mass_a_ER_retro,delta_hbar_a_ER_retro,delta_hbar0_a_ER_retro,vsurc_a_ER_retro]


if not os.path.exists('save_hbar_NS'):
    os.makedirs('save_hbar_NS')

# np.save(f'./save_hbar_NS/matrice_{n}.npy',all_a)
# np.save(f'./save_hbar_NS/matrice_{n}_GR.npy',all_a_GR)
# np.save(f'./save_hbar_NS/matrice_{n}_ER_retro.npy',all_a_ER_retro)

all_a = np.load(f'save_hbar_NS/matrice_{n}.npy')
print(all_a)
size_a = all_a[0]
mass_a = all_a[1]
delta_hbar_a = all_a[2]
delta_hbar0_a = all_a[3]
vsurc_a = all_a[4]

all_a_GR = np.load(f'save_hbar_NS/matrice_{n}_GR.npy')

size_a_GR = all_a_GR[0]
mass_a_GR = all_a_GR[1]
delta_hbar_a_GR = all_a_GR[2]
vsurc_a_GR = all_a_GR[3]

all_a_ER_retro = np.load(f'save_hbar_NS/matrice_{n}_ER_retro.npy')

size_a_ER_retro = all_a_ER_retro[0]
mass_a_ER_retro = all_a_ER_retro[1]
delta_hbar_a_ER_retro = all_a_ER_retro[2]
vsurc_a_ER_retro = all_a_ER_retro[3]

index_max = np.where(delta_hbar_a == np.max(delta_hbar_a))[0][0]
index_min = np.where(delta_hbar_a == np.min(delta_hbar_a))[0][0]


index_max_ER_retro = np.where(delta_hbar_a_ER_retro == np.max(delta_hbar_a_ER_retro))[0][0]
index_min_ER_retro = np.where(delta_hbar_a_ER_retro == np.min(delta_hbar_a_ER_retro))[0][0]


# if np.abs(np.max(delta_hbar_a)) > np.abs(np.min(delta_hbar_a)):
#     print(f'The (absolute) maximal relative difference in hbar is {np.max(delta_hbar_a) * 1e2:.1f} %\nMass =  {mass_a[index_max]:.1f} Solar mass \nRadius = {size_a[index_max]:.1f} km ')
#         ### POUR LE MAX ####
#     key = 'MAX_delta_hbar_abs_ER'
#     # value = np.max(delta_hbar_a) * 1e2
#     # value = f"{value:.1e}"
#
#     # value = "{:.2e}".format(value)
#     value = "{:.2e}".format(np.max(delta_hbar_a) * 1e2)
#     base, exponent = value.split("e")
#     exponent = int(exponent)
#     value_latex = f"{base}" #    value_latex = f"{base} \\times 10^{{{exponent}}}"
#     save_var_latex(key, value_latex)
#
#     key = 'associated_mass_ER'
#     # value = mass_a[index_max]
#     # value = f"{value:.1e}"
#     value = "{:.2e}".format(mass_a[index_max])
#     base, exponent = value.split("e")
#     exponent = int(exponent)
#     value_latex = f"{base}"
#     save_var_latex(key, value_latex)
#
#     key = 'associated_radius_ER'
#     value = "{:.1f}".format(size_a[index_max])
#     # base, exponent = value.split("e")
#     # exponent = int(exponent)
#     # value_latex = f"{base} \\times 10^{{{exponent}}}"
#     save_var_latex(key, value)
#
#     print(f'The (absolute) minimal relative difference in hbar is {np.min(delta_hbar_a) * 1e2:.1f} %\nMass =  {mass_a[index_min]:.1f} Solar mass \nRadius = {size_a[index_min]:.1f} km ')
#
#     ### POUR LE MIN ####
#     key = 'MIN_delta_hbar_abs_ER'
#     value = "{:.2e}".format(np.min(delta_hbar_a) * 1e2)
#     base, exponent = value.split("e")
#     exponent = int(exponent)
#     value_latex = f"{base}" #    value_latex = f"{base} \\times 10^{{{exponent}}}"
#     save_var_latex(key, value_latex)
#
#     key = 'associated_mass_ER'
#     value = "{:.2e}".format(mass_a[index_min])
#     base, exponent = value.split("e")
#     exponent = int(exponent)
#     value_latex = f"{base}"
#     save_var_latex(key, value_latex)
#
#     key = 'associated_radius_ER'
#     value = "{:.1f}".format(size_a[index_min])
#     save_var_latex(key, value)
#
# else:
#     print(f'The (absolute) maximal relative difference in hbar is {np.min(delta_hbar_a) * 1e2:.1f} %\nMass =  {mass_a[index_min]:.1f} Solar mass \nRadius = {size_a[index_min]:.1f} km ')
#         ### POUR LE MAX ####
#     key = 'MAX_delta_hbar_abs_ER'
#     value = "{:.2e}".format(np.min(delta_hbar_a) * 1e2)
#     base, exponent = value.split("e")
#     exponent = int(exponent)
#     value_latex = f"{base}" #    value_latex = f"{base} \\times 10^{{{exponent}}}"
#     save_var_latex(key, value_latex)
#
#     key = 'associated_mass_ER'
#     value = "{:.2e}".format(mass_a[index_min])
#     base, exponent = value.split("e")
#     exponent = int(exponent)
#     value_latex = f"{base}"
#     save_var_latex(key, value_latex)
#
#     key = 'associated_radius_ER'
#     value = "{:.1f}".format(size_a[index_min])
#     save_var_latex(key, value)
#
#     print(f'The (absolute) minimal relative difference in hbar is {np.max(delta_hbar_a) * 1e2:.1f} %\nMass =  {mass_a[index_max]:.1f} Solar mass \nRadius = {size_a[index_max]:.1f} km ')
#     ### POUR LE MIN ####
#
#     key = 'MAX_delta_hbar_abs_ER'
#     value = "{:.2e}".format(np.max(delta_hbar_a) * 1e2)
#     base, exponent = value.split("e")
#     exponent = int(exponent)
#     value_latex = f"{base}"
#     save_var_latex(key, value_latex)
#
#     key = 'associated_mass_ER'
#     value = "{:.2e}".format(mass_a[index_max])
#     base, exponent = value.split("e")
#     exponent = int(exponent)
#     value_latex = f"{base}"
#     save_var_latex(key, value_latex)
#
#     key = 'associated_radius_ER'
#     value = "{:.1f}".format(size_a[index_max])
#     save_var_latex(key, value)
#
#
# if np.abs(np.max(delta_hbar_a_ER_retro)) > np.abs(np.min(delta_hbar_a_ER_retro)):
#     print(f'With retroaction, the (absolute) maximal relative difference in hbar is {np.max(delta_hbar_a_ER_retro) * 1e2:.1f} %\nMass =  {mass_a_ER_retro[index_max_ER_retro]:.1f} Solar mass \nRadius = {size_a_ER_retro[index_max_ER_retro]:.1f} km ')
#
#         ### POUR LE MAX_ER_retro ####
#     key = 'MAX_delta_hbar_abs_ER_retro'
#     value = "{:.2e}".format(np.max(delta_hbar_a_ER_retro) * 1e2)
#     base, exponent = value.split("e")
#     exponent = int(exponent)
#     value_latex = f"{base}"
#     save_var_latex(key, value_latex)
#
#     key = 'associated_mass_ER_retro'
#     value = "{:.2e}".format(mass_a_ER_retro[index_max_ER_retro])
#     exponent = int(exponent)
#     value_latex = f"{base}"
#     save_var_latex(key, value_latex)
#
#     key = 'associated_radius_ER_retro'
#     value = "{:.1f}".format(size_a_ER_retro[index_max_ER_retro])
#     save_var_latex(key, value)
#
#
#
#     print(f'With retroaction, the (absolute) minimal relative difference in hbar is {np.min(delta_hbar_a_ER_retro) * 1e2:.1f} %\nMass =  {mass_a_ER_retro[index_min_ER_retro]:.1f} Solar mass \nRadius = {size_a_ER_retro[index_min_ER_retro]:.1f} km ')
#
#     ### POUR LE MIN ####
#     key = 'MIN_delta_hbar_abs_ER_retro'
#     value = "{:.2e}".format(np.min(delta_hbar_a_ER_retro) * 1e2)
#     base, exponent = value.split("e")
#     exponent = int(exponent)
#     value_latex = f"{base}"
#     save_var_latex(key, value_latex)
#
#     key = 'associated_mass_ER_retro'
#     value = "{:.2e}".format(mass_a_ER_retro[index_min_ER_retro])
#     exponent = int(exponent)
#     value_latex = f"{base}"
#     save_var_latex(key, value_latex)
#
#     key = 'associated_radius_ER_retro'
#     value = "{:.1f}".format(size_a_ER_retro[index_min_ER_retro])
#     save_var_latex(key, value)
#
#
# else:
#     print(f'With retroaction, the (absolute) maximal relative difference in hbar is {np.min(delta_hbar_a_ER_retro) * 1e2:.1f} %\nMass =  {mass_a_ER_retro[index_min_ER_retro]:.1f} Solar mass \nRadius = {size_a_ER_retro[index_min_ER_retro]:.1f} km ')
#
#         ### POUR LE MAX ####
#     key = 'MAX_delta_hbar_abs_ER_retro'
#     value = "{:.2e}".format(np.min(delta_hbar_a_ER_retro) * 1e2)
#     base, exponent = value.split("e")
#     exponent = int(exponent)
#     value_latex = f"{base}"
#     save_var_latex(key, value_latex)
#
#     key = 'associated_mass_ER_retro'
#     value = "{:.2e}".format(mass_a_ER_retro[index_min_ER_retro])
#     exponent = int(exponent)
#     value_latex = f"{base}"
#     save_var_latex(key, value_latex)
#
#     key = 'associated_radius_ER_retro'
#     value = "{:.1f}".format(size_a_ER_retro[index_min_ER_retro])
#     save_var_latex(key, value)
#
#     print(f'With retroaction, the (absolute) minimal relative difference in hbar is {np.max(delta_hbar_a_ER_retro) * 1e2:.1f} %\nMass =  {mass_a_ER_retro[index_max_ER_retro]:.1f} Solar mass \nRadius = {size_a_ER_retro[index_max_ER_retro]:.1f} km ')
#
#     ### POUR LE MIN ####
#     key = 'MIN_delta_hbar_abs_ER_retro'
#     value = "{:.2e}".format(np.max(delta_hbar_a_ER_retro) * 1e2)
#     base, exponent = value.split("e")
#     exponent = int(exponent)
#     value_latex = f"{base}"
#     save_var_latex(key, value_latex)
#
#     key = 'associated_mass_ER_retro'
#     value = "{:.2e}".format(mass_a_ER_retro[index_max_ER_retro])
#     exponent = int(exponent)
#     base, exponent = value.split("e")
#     value_latex = f"{base}"
#     save_var_latex(key, value_latex)
#
#     key = 'associated_radius_ER_retro'
#     value = "{:.1f}".format(size_a_ER_retro[index_max_ER_retro])
#     save_var_latex(key, value)
#





index_max_M = np.where(mass_a == np.max(mass_a))[0][0]
print(f'The relative variation of hbar for the most massive NS of {mass_a[index_max_M]:.1f} M_SUN and a radius of {size_a[index_max_M]:.1f} km is {delta_hbar_a[index_max_M] * 1e2:.1f} %')

key = 'delta_hbar_relative_most_massive_NS_ER'
value = "{:.2e}".format(delta_hbar_a[index_max_M] * 1e2)
base, exponent = value.split("e")
exponent = int(exponent)
value_latex = f"{base}"
save_var_latex(key, value_latex)

key = 'Most_massive_star_ER'
value = "{:.2f}".format(mass_a[index_max_M])
# exponent = int(exponent)
# base, exponent = value.split("e")
# value_latex = f"{base}"
save_var_latex(key, value)

key = 'associated_radius_ER'
value = "{:.1f}".format(size_a[index_max_M])
save_var_latex(key, value)


key = 'associated_density_ER'
value = den_space[index_max_M]
value = f"{value:.0f}"
save_var_latex(key, value)

index_max_M_ER_retro = np.where(mass_a_ER_retro == np.max(mass_a_ER_retro))[0][0]
print(f'The relative variation of hbar for the most massive NS with retroaction of {mass_a_ER_retro[index_max_M_ER_retro]:.1f} M_SUN and a radius of {size_a_ER_retro[index_max_M_ER_retro]:.1f} km is {delta_hbar_a_ER_retro[index_max_M_ER_retro] * 1e2:.1f} %')

#### La valeur de masse max donnée et celle de la densité max dépasse le treshold donc c'st un probleme, il faut poser un masque avant. Je vais donc d'abord poser un masque qui ne prend que les valeurs de masse avant treshold et ensuite prendre la masse max de ce masque


# key = 'delta_hbar_relative_most_massive_NS_ER_retro'
# value = "{:.2e}".format(delta_hbar_a_ER_retro[index_max_M_ER_retro] * 1e2)
# exponent = int(exponent)
# base, exponent = value.split("e")
# value_latex = f"{base}"
# save_var_latex(key, value_latex)

R_tresh_ER_retro = all_a_ER_retro[0][all_a_ER_retro[4] < 1/ np.sqrt(3)]
M_tresh_ER_retro = all_a_ER_retro[1][all_a_ER_retro[4] < 1/ np.sqrt(3)]

index_most_massive_ER_retro = np.where(M_tresh_ER_retro == np.max(M_tresh_ER_retro))[0][0]

key = 'Most_massive_star_ER_retro'
value = "{:.2e}".format(mass_a_ER_retro[index_most_massive_ER_retro])
exponent = int(exponent)
base, exponent = value.split("e")
value_latex = f"{base}"
save_var_latex(key, value_latex)

key = 'associated_radius_ER_retro'
value = "{:.1f}".format(size_a_ER_retro[index_most_massive_ER_retro])
save_var_latex(key, value)

key = 'associated_density_ER_retro'
# prout
value = den_space[index_most_massive_ER_retro]
value = f"{value:.0f}"
save_var_latex(key, value)


index_max_M_GR = np.where(mass_a_GR == np.max(mass_a_GR))[0][0]
print(f'The relative variation of hbar for the most massive NS of {mass_a_GR[index_max_M_GR]:.1f} M_SUN and a radius of {size_a_GR[index_max_M_GR]:.1f} km is {delta_hbar_a_GR[index_max_M_GR] * 1e2:.1f} %')

# key = 'delta_hbar_relative_most_massive_NS_GR'
# value = delta_hbar_a_GR[index_max_M_GR] * 1e2
# value = f"{value:.1e}"
# save_var_latex(key, value)
#
key = 'Most_massive_star_GR'
value = mass_a_GR[index_max_M_GR]
value = f"{value:.2f}"
save_var_latex(key, value)

key = 'associated_radius_GR'
value = size_a_GR[index_max_M_GR]
value = f"{value:.1f}"
save_var_latex(key, value)

key = 'associated_density_GR'
value = den_space[index_max_M_GR]
value = f"{value:.0f}"
save_var_latex(key, value)


R_tresh = all_a[0][all_a[4] < 1/ np.sqrt(3)]
M_tresh = all_a[1][all_a[4] < 1/ np.sqrt(3)]
delta_h_tresh = all_a[2][all_a[4] < 1/ np.sqrt(3)]
delta_h_central_tresh = all_a[3][all_a[4] < 1/ np.sqrt(3)]


R_tresh_GR = all_a_GR[0][all_a_GR[3] < 1/ np.sqrt(3)]
M_tresh_GR = all_a_GR[1][all_a_GR[3] < 1/ np.sqrt(3)]
delta_h_tresh_GR = all_a_GR[2][all_a_GR[3] < 1/ np.sqrt(3)]


R_tresh_ER_retro = all_a_ER_retro[0][all_a_ER_retro[4] < 1/ np.sqrt(3)]
M_tresh_ER_retro = all_a_ER_retro[1][all_a_ER_retro[4] < 1/ np.sqrt(3)]
delta_h_tresh_ER_retro = all_a_ER_retro[2][all_a_ER_retro[4] < 1/ np.sqrt(3)]
delta_h_central_tresh_ER_retro = all_a_ER_retro[3][all_a_ER_retro[4] < 1/ np.sqrt(3)]


dens_tresh = den_space[np.where(all_a[4] < 1/ np.sqrt(3))[0][-1]]
print(f'In ER, above the density of {dens_tresh:.0f} Mev/fm^3, the speed of sound somewhere inside the neutron star becomes larger than the conservative limit c/sqrt(3)\n')

index_treshold_ER = np.where(all_a[4] < 1/ np.sqrt(3))[0][-1]


key = 'density_treshold_ER'
value = f"{dens_tresh:.0f}"
save_var_latex(key, value)

key = 'mass_density_treshold_ER'
value = "{:.2e}".format(M_tresh[index_treshold_ER])
exponent = int(exponent)
base, exponent = value.split("e")
value_latex = f"{base}"
save_var_latex(key, value_latex)


key = 'radius_density_treshold_ER'
value = "{:.1f}".format(R_tresh[index_treshold_ER])
save_var_latex(key, value)

index_treshold_GR = np.where(all_a_GR[3] < 1/ np.sqrt(3))[0][-1]

dens_tresh_GR = den_space[np.where(all_a_GR[3] < 1/ np.sqrt(3))[0][-1]]
print(f'In GR, above the density of {dens_tresh_GR:.0f} Mev/fm^3, the speed of sound somewhere inside the neutron star becomes larger than the conservative limit c/sqrt(3)')

key = 'density_treshold_GR'
value = f"{dens_tresh_GR:.0f}"
# exponent = int(exponent)
# base, exponent = value.split("e")
# value_latex = f"{base}"
save_var_latex(key, value)

key = 'mass_density_treshold_GR'
value = "{:.2e}".format(M_tresh_GR[index_treshold_GR])
exponent = int(exponent)
base, exponent = value.split("e")
value_latex = f"{base}"
save_var_latex(key, value_latex)

key = 'radius_density_treshold_GR'
value = "{:.1f}".format(R_tresh_GR[index_treshold_GR])
save_var_latex(key, value)

index_treshold_ER_retro = np.where(all_a_ER_retro[4] < 1/ np.sqrt(3))[0][-1]
dens_tresh_ER_retro = den_space[np.where(all_a_ER_retro[4] < 1/ np.sqrt(3))[0][-1]]
print(f'In ER_retro, above the density of {dens_tresh_ER_retro:.0f} Mev/fm^3, the speed of sound somewhere inside the neutron star becomes larger than the conservative limit c/sqrt(3)\n')

key = 'density_treshold_ER_retro'
# value = "{:.2e}".format(dens_tresh_ER_retro)
# exponent = int(exponent)
# base, exponent = value.split("e")
# value_latex = f"{base}"
value = f"{dens_tresh_ER_retro:.0f}"
save_var_latex(key, value)

key = 'mass_density_treshold_ER_retro'
print(M_tresh_ER_retro[-1])
value = "{:.2e}".format(M_tresh_ER_retro[index_treshold_ER_retro])
exponent = int(exponent)
base, exponent = value.split("e")
value_latex = f"{base}"
save_var_latex(key, value_latex)

key = 'radius_density_treshold_ER_retro'
value = "{:.1f}".format(R_tresh_ER_retro[index_treshold_ER_retro])
save_var_latex(key, value)



### Recover mass adm and density max :

M_tresh = list(M_tresh)
M_tresh_GR = list(M_tresh_GR)
M_tresh_ER_retro = list(M_tresh_ER_retro)

Max_mass_ADM = max(M_tresh)
Max_mass_ADM_index = M_tresh.index(Max_mass_ADM)
Max_density_ADM = den_space[Max_mass_ADM_index]#/(cst.eV*10**6/(cst.c**2*cst.fermi**3))


Max_mass_ADM_GR = max(M_tresh_GR)
Max_mass_ADM_GR_index = M_tresh_GR.index(Max_mass_ADM_GR)
Max_density_ADM_GR = den_space[Max_mass_ADM_GR_index]#/(cst.eV*10**6/(cst.c**2*cst.fermi**3))



Max_mass_ADM_ER_retro = max(M_tresh_ER_retro)
Max_mass_ADM_ER_retro_index = M_tresh_ER_retro.index(Max_mass_ADM_ER_retro)
Max_density_ADM_ER_retro = den_space[Max_mass_ADM_ER_retro_index]#/(cst.eV*10**6/(cst.c**2*cst.fermi**3))

print('Le maxmimu de masse ADM - densité est de : ')
print( 'GR = ', Max_mass_ADM_GR, 'kg', 'et', Max_density_ADM_GR , 'Mev/fm3')
print('ER =', Max_mass_ADM, 'kg', 'ET', Max_density_ADM, 'Mev/fm3')
print('ER_retro = ' , Max_mass_ADM_ER_retro,'kg', 'et', Max_density_ADM_ER_retro , 'Mev/fm3')



##################################################################
## Ici sera le plot mass max vs radius et delta hbar surface

# plt.scatter(all_a[0],all_a[1], c = all_a[2], marker = 's', cmap = 'Blues_r')
plt.figure()
plt.scatter(R_tresh,M_tresh, c = delta_h_tresh * 1e2, marker = 's', cmap = 'gray')
plt.plot(R_tresh_GR ,M_tresh_GR, label='GR', color='tab:gray', linestyle='dashed')
plt.legend()
plt.xlabel('Radius (km)')
plt.ylabel('Mass (M\u2609)')
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.8, 0.1, 0.055, 0.8])
plt.colorbar(cax=cax).set_label('$\delta \hbar / \hbar (\%)$')
plt.savefig(f'./save_hbar_NS/scatter_treshold_{n}.png', dpi= 200, bbox_inches="tight")
# plt.show()
plt.close()



index_max = np.where(delta_h_tresh == np.max(delta_h_tresh))[0][0]
index_min = np.where(delta_h_tresh == np.min(delta_h_tresh))[0][0]

if np.abs(np.max(delta_h_tresh)) > np.abs(np.min(delta_h_tresh)):
    print(f'The (absolute) maximal relative difference in hbar is {np.max(delta_h_tresh) * 1e2:.1f} %\nMass =  {M_tresh[index_max]:.1f} Solar mass \nRadius = {R_tresh[index_max]:.1f} km ')


    key = 'MAX_abs_rel_diff_hbar_ER'
    value = f"{np.max(delta_h_tresh) * 1e2:.1f}"
    # exponent = int(exponent)
    # base, exponent = value.split("e")
    # value_latex = f"{base}"
    save_var_latex(key, value)

    key = 'MAX_abs_rel_diff_mass_ER'
    value = f"{M_tresh[index_max]:.1f}"
    # exponent = int(exponent)
    # base, exponent = value.split("e")
    # value_latex = f"{base}"
    save_var_latex(key, value)

    key = 'MAX_abs_rel_diff_radius_ER'
    value = "{:.1f}".format(R_tresh[index_max])
    save_var_latex(key, value)

    print(f'The (absolute) minimal relative difference in hbar is {np.min(delta_h_tresh) * 1e2:.1f} %\nMass =  {M_tresh[index_min]:.1f} Solar mass \nRadius = {R_tresh[index_min]:.1f} km ')

    key = 'MIN_abs_rel_diff_hbar_ER'
    # value = "{:.2e}".format(np.min(delta_h_tresh) * 1e2)
    value = f'{np.min(delta_h_tresh) * 1e2:.1f}'
    # exponent = int(exponent)
    # base, exponent = value.split("e")
    # value_latex = f"{base}"
    save_var_latex(key, value)

    key = 'MIN_abs_rel_diff_mass_ER'
    value = f"{M_tresh[index_min]:.1f}"
    # exponent = int(exponent)
    # base, exponent = value.split("e")
    # value_latex = f"{base}"
    save_var_latex(key, value)

    key = 'MIN_abs_rel_diff_radius_ER'
    value = "{:.1f}".format(R_tresh[index_min])
    save_var_latex(key, value)


else:
    print(f'The (absolute) maximal relative difference in hbar is {np.min(delta_h_tresh) * 1e2:.1f} %\nMass =  {M_tresh[index_min]:.1f} Solar mass \nRadius = {R_tresh[index_min]:.1f} km ')
    key = 'MAX_abs_rel_diff_hbar_ER'
    value = "{:.2e}".format(np.min(delta_h_tresh) * 1e2)
    exponent = int(exponent)
    base, exponent = value.split("e")
    value_latex = f"{base}"
    save_var_latex(key, value_latex)

    key = 'MAX_abs_rel_diff_mass_ER'
    print(M_tresh_ER_retro[-1])
    value = "{:.2e}".format(M_tresh[index_min])
    exponent = int(exponent)
    base, exponent = value.split("e")
    value_latex = f"{base}"
    save_var_latex(key, value_latex)

    key = 'MAX_abs_rel_diff_radius_ER'
    value = "{:.1f}".format(R_tresh[index_min])
    save_var_latex(key, value)

    print(f'The (absolute) minimal relative difference in hbar is {np.max(delta_h_tresh) * 1e2:.1f} %\nMass =  {M_tresh[index_max]:.1f} Solar mass \nRadius = {R_tresh[index_max]:.1f} km ')

    key = 'MIN_abs_rel_diff_hbar_ER'
    value = "{:.2e}".format(np.max(delta_h_tresh) * 1e2)
    exponent = int(exponent)
    base, exponent = value.split("e")
    value_latex = f"{base}"
    save_var_latex(key, value_latex)

    key = 'MIN_abs_rel_diff_mass_ER'
    print(M_tresh_ER_retro[-1])
    value = "{:.2e}".format(M_tresh[index_max])
    exponent = int(exponent)
    base, exponent = value.split("e")
    value_latex = f"{base}"
    save_var_latex(key, value_latex)

    key = 'MIN_abs_rel_diff_radius_ER'
    value = "{:.1f}".format(R_tresh[index_max])
    save_var_latex(key, value)




#######################################################################



##### Ici sera le plot mass max vs radius et delta hbar center
plt.figure()
plt.scatter(R_tresh,M_tresh, c = delta_h_central_tresh * 1e2, marker = 's', cmap = 'gray')
plt.plot(R_tresh_GR ,M_tresh_GR, label='GR', color='tab:gray', linestyle='dashed')
plt.legend()
plt.xlabel('Radius (km)')
plt.ylabel('Mass (M\u2609)')
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.8, 0.1, 0.055, 0.8])
plt.colorbar(cax=cax).set_label('$\delta \hbar / \hbar (\%)$')
plt.savefig(f'./save_hbar_NS/scatter_1000.png', dpi= 200, bbox_inches="tight")
# plt.show()
plt.close()



index_max = np.where(delta_h_central_tresh == np.max(delta_h_central_tresh))[0][0]
index_min = np.where(delta_h_central_tresh == np.min(delta_h_central_tresh))[0][0]

if np.abs(np.max(delta_h_central_tresh)) > np.abs(np.min(delta_h_central_tresh)):
    print(f'The (absolute) maximal relative difference in hbar is {np.max(delta_h_central_tresh) * 1e2:.1f} %\nMass =  {M_tresh[index_max]:.1f} Solar mass \nRadius = {R_tresh[index_max]:.1f} km ')


    key = 'MAX_abs_rel_diff_central_hbar_ER'
    value = f"{np.max(delta_h_central_tresh) * 1e2:.1f}"
    # exponent = int(exponent)
    # base, exponent = value.split("e")
    # value_latex = f"{base}"
    save_var_latex(key, value)

    key = 'MAX_abs_rel_diff_central_mass_ER'
    value = f"{M_tresh[index_max]:.1f}"
    # exponent = int(exponent)
    # base, exponent = value.split("e")
    # value_latex = f"{base}"
    save_var_latex(key, value)

    key = 'MAX_abs_rel_diff_central_radius_ER'
    value = "{:.1f}".format(R_tresh[index_max])
    save_var_latex(key, value)

    print(f'The (absolute) minimal relative difference in hbar is {np.min(delta_h_central_tresh) * 1e2:.1f} %\nMass =  {M_tresh[index_min]:.1f} Solar mass \nRadius = {R_tresh[index_min]:.1f} km ')


    key = 'MIN_abs_rel_diff_central_hbar_ER'
    # value = "{:.2e}".format(np.min(delta_h_tresh) * 1e2)
    value = f'{np.min(delta_h_central_tresh) * 1e2:.1f}'
    # exponent = int(exponent)
    # base, exponent = value.split("e")
    # value_latex = f"{base}"
    save_var_latex(key, value)

    key = 'MIN_abs_rel_diff_central_mass_ER'
    value = f"{M_tresh[index_min]:.1f}"
    # exponent = int(exponent)
    # base, exponent = value.split("e")
    # value_latex = f"{base}"
    save_var_latex(key, value)

    key = 'MIN_abs_rel_diff_central_radius_ER'
    value = "{:.1f}".format(R_tresh[index_min])
    save_var_latex(key, value)

else:
    print(f'The (absolute) maximal relative difference in hbar is {np.min(delta_h_central_tresh) * 1e2:.1f} %\nMass =  {M_tresh[index_min]:.1f} Solar mass \nRadius = {R_tresh[index_min]:.1f} km ')

    key = 'MAX_abs_rel_diff_central_hbar_ER'
    # value = "{:.2e}".format(np.min(delta_h_tresh) * 1e2)
    value = f'{np.min(delta_h_central_tresh) * 1e2:.1f}'
    # exponent = int(exponent)
    # base, exponent = value.split("e")
    # value_latex = f"{base}"
    save_var_latex(key, value)

    key = 'MAX_abs_rel_diff_central_mass_ER'
    value = f"{M_tresh[index_min]:.1f}"
    # exponent = int(exponent)
    # base, exponent = value.split("e")
    # value_latex = f"{base}"
    save_var_latex(key, value)

    key = 'MAX_abs_rel_diff_central_radius_ER'
    value = "{:.1f}".format(R_tresh[index_min])
    save_var_latex(key, value)



    print(f'The (absolute) minimal relative difference in hbar is {np.max(delta_h_central_tresh) * 1e2:.1f} %\nMass =  {M_tresh[index_max]:.1f} Solar mass \nRadius = {R_tresh[index_max]:.1f} km ')



    key = 'MIN_abs_rel_diff_central_hbar_ER'
    value = f"{np.max(delta_h_central_tresh) * 1e2:.1f}"
    # exponent = int(exponent)
    # base, exponent = value.split("e")
    # value_latex = f"{base}"
    save_var_latex(key, value)

    key = 'MIN_abs_rel_diff_central_mass_ER'
    value = f"{M_tresh[index_max]:.1f}"
    # exponent = int(exponent)
    # base, exponent = value.split("e")
    # value_latex = f"{base}"
    save_var_latex(key, value)

    key = 'MIN_abs_rel_diff_central_radius_ER'
    value = "{:.1f}".format(R_tresh[index_max])
    save_var_latex(key, value)


########### TEST PLOT RETRO ##########################

# ##### Ici sera le plot mass max vs radius et delta hbar center
# plt.figure()
# plt.scatter(R_tresh_ER_retro,M_tresh_ER_retro, c = delta_h_central_tresh_ER_retro * 1e2, marker = 's', cmap = 'gray')
# plt.plot(R_tresh_GR ,M_tresh_GR, label='GR', color='tab:gray', linestyle='dashed')
# plt.legend()
# plt.xlabel('Radius (km)')
# plt.ylabel('Mass (M\u2609)')
# plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
# cax = plt.axes([0.8, 0.1, 0.055, 0.8])
# plt.colorbar(cax=cax).set_label('$\delta \hbar / \hbar (\%)$')
# plt.savefig(f'./save_hbar_NS/scatter_1000_ER_retro.png', dpi= 200, bbox_inches="tight")
# # plt.show()
# plt.close()


############################################ PLOT 3 ######################################


def hbar_effect(initDensity):
    dilaton = True
    retro = False
    log_active = False
    dependence = 2
    PhiInit = 1
    PsiInit = 0
    option = 1
    radiusMax_in = 40000#50000
    radiusMax_out = 100000000
    Npoint = 50000
    initDensity *= cst.eV*10**6/(cst.c**2*cst.fermi**3)
    tov = TOV(initDensity , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton, log_active,retro, dependence)
    tov.ComputeTOV_normalization()
    radius = tov.radius
    hbar = tov.hbar
    radius_star = tov.radiusStar

    return hbar, radius, radius_star



def hbar_effect_retro(initDensity):
    dilaton = True
    log_active = False
    retro = True
    dependence = 2
    PhiInit = 1
    PsiInit = 0
    option = 1
    radiusMax_in = 40000#50000
    radiusMax_out = 100000000
    Npoint = 50000
    initDensity *= cst.eV*10**6/(cst.c**2*cst.fermi**3)

    tov = TOV(initDensity , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton, log_active,retro, dependence)
    PhiInit = tov.find_dilaton_center()[0]
    tov = TOV(initDensity , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton, log_active,retro, dependence)
    tov.ComputeTOV()
    hbar_retro = tov.hbar
    radius_retro = tov.radius

    return hbar_retro, radius_retro


all_normal = np.load(f'./save_hbar_NS/matrice_normal.npy')

all_retro = np.load(f'./save_hbar_NS/matrice_retro.npy')

radius_Star = np.load(f'./save_hbar_NS/matrice_radius_star.npy')

# hbar_values, radius_values, radius_star_value = hbar_effect(1000)
# hbar_values_retro, radius_values_retro = hbar_effect_retro(1000)

# all_retro = [hbar_values_retro, radius_values_retro]
# all_normal = [hbar_values, radius_values]
# radius_Star = [radius_star_value]


# np.save(f'./save_hbar_NS/matrice_normal.npy',all_normal)
# np.save(f'./save_hbar_NS/matrice_radius_star.npy',radius_Star)
# np.save(f'./save_hbar_NS/matrice_retro.npy',all_retro)


hbar_normal = all_normal[0]
hbar_retro = all_retro[0]
radius_normal = all_normal[1]/1e3
radius_retro = all_retro[1]/1e3
star_radius= radius_Star[0]/1e3

#Plot
plt.figure()
plt.plot(radius_retro, hbar_retro, label = 'With retroaction')
plt.plot(radius_normal, hbar_normal, label = 'Without retroaction' )
plt.xlim(-2, 60)
plt.ylim(0.999,1.045)
plt.axvline(star_radius, color='r', linestyle='--', label='Star radius')
if len(radius_retro) < len(radius_normal):
    plt.fill_between(radius_normal[:len(radius_retro)], hbar_normal[:len(hbar_retro)], hbar_retro, where=(hbar_retro > hbar_normal[:len(hbar_retro)]), color='lightgray', alpha=0.5)
else:
    plt.fill_between(radius_retro[:len(radius_normal)], hbar_retro[:len(hbar_normal)], hbar_normal, where=(hbar_retro[:len(hbar_normal)] > hbar_normal), color='lightgray', alpha=0.5)

plt.xlabel('Distance (km)', fontsize=19)
plt.ylabel(r'$\hbar$ variation', fontsize=22)
plt.legend()
plt.savefig('./save_hbar_NS/hbar_variation_comparison_NS')
# plt.show()






############# Obtenir les valeurs en fonction de la dépendance en hbar ##########





def hbar_dependence_effect(initDensity):

    retro = True
    dilaton = True
    log_active = False
    PhiInit = 1
    PsiInit = 0
    option = 1
    radiusMax_in = 40000#50000
    radiusMax_out = 100000000
    Npoint = 50000
    initDensity *= cst.eV*10**6/(cst.c**2*cst.fermi**3)
    #
    # tov = TOV(initDensity , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton, log_active,retro, 0)
    # PhiInit = tov.find_dilaton_center()[0]
    # tov = TOV(initDensity , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton, log_active,retro, 0)
    # tov.ComputeTOV()
    # tov.hbar_into_txt()
    # tov.radius_into_txt()
    # tov.hbar_star_txt()

    tov = TOV(initDensity , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton, log_active,retro, dependence)
    PhiInit = tov.find_dilaton_center()[0]
    tov = TOV(initDensity , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton, log_active,retro, dependence)
    tov.ComputeTOV()
    hbar = tov.hbar
    radius = tov.radius
    hbar_star = tov.hbar_star
    return hbar, radius, hbar_star


Dependence = []
for i in range(-3 ,4):
    Dependence.append(i)



# for dependence in tqdm(Dependence):
#     if dependence == -3:
#         hbar_value_m3, radius_value_m3, hbar_star_value_m3 = hbar_dependence_effect(1000)
#     elif dependence == -2:
#         hbar_value_m2, radius_value_m2, hbar_star_value_m2 = hbar_dependence_effect(1000)
#     elif dependence == -1:
#         hbar_value_m1, radius_value_m1, hbar_star_value_m1 = hbar_dependence_effect(1000)
#     elif dependence == 0:
#         hbar_value, radius_value, hbar_star_value = hbar_dependence_effect(1000)
#     elif dependence == 1:
#         hbar_value_1, radius_value_1, hbar_star_value_1 = hbar_dependence_effect(1000)
#     elif dependence == 2:
#         hbar_value_2, radius_value_2, hbar_star_value_2 = hbar_dependence_effect(1000)
#     elif dependence == 3:
#         hbar_value_3, radius_value_3, hbar_star_value_3 = hbar_dependence_effect(1000)


all_m3 = np.load(f'./save_hbar_NS/matrice_m3.npy')
all_m2 = np.load(f'./save_hbar_NS/matrice_m2.npy')
all_m1 = np.load(f'./save_hbar_NS/matrice_m1.npy')
all_0 = np.load(f'./save_hbar_NS/matrice_0.npy')
all_1 = np.load(f'./save_hbar_NS/matrice_1.npy')
all_2 = np.load(f'./save_hbar_NS/matrice_2.npy')
all_3 = np.load(f'./save_hbar_NS/matrice_3.npy')


all_star_m3 = np.load(f'./save_hbar_NS/matrice_star_m3.npy')
all_star_m2 = np.load(f'./save_hbar_NS/matrice_star_m2.npy')
all_star_m1 = np.load(f'./save_hbar_NS/matrice_star_m1.npy')
all_star_0 = np.load(f'./save_hbar_NS/matrice_star_0.npy')
all_star_1 = np.load(f'./save_hbar_NS/matrice_star_1.npy')
all_star_2 = np.load(f'./save_hbar_NS/matrice_star_2.npy')
all_star_3 = np.load(f'./save_hbar_NS/matrice_star_m3.npy')


# all_m3 = [hbar_value_m3]
# all_m2 = [hbar_value_m2]
# all_m1 = [hbar_value_m1]
# all_0 = [hbar_value]
# all_1 = [hbar_value_1]
# all_2 = [hbar_value_2]
# all_3 = [hbar_value_3]


#
# all_star_m3 = [hbar_star_value_m3]
# all_star_m2 = [hbar_star_value_m2]
# all_star_m1 = [hbar_star_value_m1]
# all_star_0 = [hbar_star_value]
# all_star_1 = [hbar_star_value_1]
# all_star_2 = [hbar_star_value_2]
# all_star_3 = [hbar_star_value_3]
#


#
# np.save(f'./save_hbar_NS/matrice_m3.npy',all_m3)
# np.save(f'./save_hbar_NS/matrice_m2.npy',all_m2)
# np.save(f'./save_hbar_NS/matrice_m1.npy',all_m1)
# np.save(f'./save_hbar_NS/matrice_0.npy',all_0)
# np.save(f'./save_hbar_NS/matrice_1.npy',all_1)
# np.save(f'./save_hbar_NS/matrice_2.npy',all_2)
# np.save(f'./save_hbar_NS/matrice_3.npy',all_3)
#
# np.save(f'./save_hbar_NS/matrice_star_m3.npy',all_star_m3)
# np.save(f'./save_hbar_NS/matrice_star_m2.npy',all_star_m2)
# np.save(f'./save_hbar_NS/matrice_star_m1.npy',all_star_m1)
# np.save(f'./save_hbar_NS/matrice_star_0.npy',all_star_0)
# np.save(f'./save_hbar_NS/matrice_star_1.npy',all_star_1)
# np.save(f'./save_hbar_NS/matrice_star_2.npy',all_star_2)
# np.save(f'./save_hbar_NS/matrice_star_3.npy',all_star_3)
#









### surface ####
hbar_inf = 1

key = 'hbar_error_-3'
value = (all_star_m3[0] - all_star_0[0])/(all_star_m3[0]-hbar_inf) * 100
value = f"{value:.0f}"
save_var_latex_dependence(key, value)

key = 'hbar_error_-2'
value = (all_star_m2[0] - all_star_0[0])/(all_star_m2[0]-hbar_inf) * 100
value = f"{value:.0f}"
save_var_latex_dependence(key, value)

key = 'hbar_error_-1'
value = (all_star_m1[0] - all_star_0[0])/(all_star_m1[0]-hbar_inf) * 100
value = f"{value:.0f}"
save_var_latex_dependence(key, value)

key = 'hbar_error_1'
value = (all_star_1[0] - all_star_0[0])/(all_star_1[0]-hbar_inf) * 100
value = f"{value:.0f}"
save_var_latex_dependence(key, value)

key = 'hbar_error_2'
value = (all_star_2[0] - all_star_0[0])/(all_star_2[0]-hbar_inf) * 100
value = f"{value:.0f}"
save_var_latex_dependence(key, value)

key = 'hbar_error_3'
value = (all_star_3[0] - all_star_0[0])/(all_star_3[0]-hbar_inf) * 100
value = f"{value:.0f}"
save_var_latex_dependence(key, value)


### center ####


key = 'hbar_center_error_-3'
value = (all_m3[0][0] - all_0[0][0])/(all_m3[0][0]-hbar_inf) * 100
value = f"{value:.0f}"
save_var_latex_dependence(key, value)

key = 'hbar_center_error_-2'
value = (all_m2[0][0] - all_0[0][0])/(all_m2[0][0]-hbar_inf) * 100
value = f"{value:.0f}"
save_var_latex_dependence(key, value)

key = 'hbar_center_error_-1'
value = (all_m1[0][0] - all_0[0][0])/(all_m1[0][0]-hbar_inf) * 100
value = f"{value:.0f}"
save_var_latex_dependence(key, value)

key = 'hbar_center_error_1'
value = (all_1[0][0] - all_0[0][0])/(all_1[0][0]-hbar_inf) * 100
value = f"{value:.0f}"
save_var_latex_dependence(key, value)

key = 'hbar_center_error_2'
value = (all_2[0][0] - all_0[0][0])/(all_2[0][0]-hbar_inf) * 100
value = f"{value:.0f}"
save_var_latex_dependence(key, value)

key = 'hbar_center_error_3'
value = (all_3[0][0] - all_0[0][0])/(all_3[0][0]-hbar_inf) * 100
value = f"{value:.0f}"
save_var_latex_dependence(key, value)

