from TOV import *
import matplotlib
import matplotlib.pyplot as plt
import scipy.constants as cst
import numpy as np
import os
import tqdm

c2 = cst.c**2
n = 250
lowest_density = 1e9 # kg/m3
highest_density = 1e13
densities = np.linspace(np.log(lowest_density), np.log(highest_density), n)
densities = np.exp(densities)
count = 0

def run_ER(rho_cen):
    PhiInit = 1
    PsiInit = 0
    option = 1
    radiusMax_in = 20000000
    radiusMax_out = 100000000
    Npoint = 50000
    log_active = False # Change for True for seeing star's data
    dilaton_active = True # Change for false for deactivating scalar field
    rhoInit = rho_cen#*cst.eV*10**6/(cst.c**2*cst.fermi**3)

    tov = TOV(rhoInit , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton_active, log_active, count)
    PhiInit = tov.find_dilaton_center()[0]
    tov = TOV(rhoInit , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton_active, log_active, count)
    tov.ComputeTOV()

    radius_all = tov.radius
    radiusStar = tov.radiusStar
    delta_hbar_star = tov.delta_hbar_star
    delta_hbar_all = tov.delta_hbar

    return radius_all, radiusStar, delta_hbar_star, delta_hbar_all

matrices = [f'./saved_data/matrice_{n}.npy', f'./saved_data/matrice_star_{n}.npy', f'./saved_data/matrice_delta_hbar_{n}.npy']

if verify_files(matrices, n):

    all_radius_list = list(np.load(f'./saved_data/matrice_{n}.npy', allow_pickle=True))
    all_star_data = list(np.load(f'./saved_data/matrice_star_{n}.npy', allow_pickle=True))
    all_delta_hbar_list = list(np.load(f'./saved_data/matrice_delta_hbar_{n}.npy', allow_pickle=True))
    densities = densities / 1e12

    radius_star = all_star_data[0]
    delta_hbar_star = all_star_data[1]
    radius_a = all_radius_list[0]
    delta_hbar_data_a = all_delta_hbar_list[0]

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.gray_r
    adjusted_cmap = mcolors.LinearSegmentedColormap.from_list(
        'adjusted_gray_r', cmap(np.linspace(0.2, 0.8, 300)))
    colors = adjusted_cmap(densities)
    plt.plot(radius_star, delta_hbar_star,color='red', linestyle='--', label='WD surface', zorder=2)
    for i in range(len(radius_a)):
        ax.plot(radius_a[i], delta_hbar_data_a[i], color=colors[i], zorder = 1)
    norm = mcolors.Normalize(vmin=np.min(densities), vmax=np.max(densities))
    sm = plt.cm.ScalarMappable(cmap=adjusted_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=np.linspace(np.min(densities), np.max(densities), num=5))
    cbar.set_label('Core density ($10^{12}$kg/m$^3$)', fontsize=20)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    ax.set_xlabel('Distance (10$^5$km) ', fontsize=19)
    ax.set_ylabel(r'$\delta \hbar/\hbar_{\infty} $', fontsize=19)
    plt.ylim([5e-12, 4e-5])
    ax.set_yscale('log')
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.legend(loc = 'upper right')
    plt.savefig('./deltahbar_vs_radius_WD')
    # plt.show()

else:
    print('Data are not present or incomplete. Let\'s compute them')

    # Saving repository
    save_dir = 'saved_data'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Files
    all_radius = os.path.join(save_dir, f'matrice_{n}.npy')
    all_star_data = os.path.join(save_dir, f'matrice_star_{n}.npy')
    all_delta_hbar_data = os.path.join(save_dir, f'matrice_delta_hbar_{n}.npy')

    if os.path.exists(all_star_data):
        radiusStar, delta_hbarStar_all = np.load(all_star_data, allow_pickle=True)
        print(f"Founded file : {all_star_data}")
        start_idx = len(radiusStar)
    else:
        delta_hbarStar_all = np.array([])
        radiusStar = np.array([])
        start_idx = 0

    if os.path.exists(all_radius):
        radius_all_loaded = list(np.load(all_radius, allow_pickle=True))[0]
        print(f"Founded file : {all_radius}")
    else:
        radius_all_loaded = [None for _ in range(n)]

    if os.path.exists(all_delta_hbar_data):
        delta_hbar_all_loaded = list(np.load(all_delta_hbar_data, allow_pickle=True))[0]
        print(f"Founded file : {all_delta_hbar_data}")
    else:
        delta_hbar_all_loaded = [None for _ in range(n)]

    all_radius = radius_all_loaded.copy()
    all_delta_hbar = delta_hbar_all_loaded.copy()

    for i in tqdm.tqdm(range(start_idx, n)):
        den = densities[i]
        radius_e8_a, radius_e8_Star, delta_hbar_star, delta_hbar_a = run_ER(den)
        radiusStar = np.append(radiusStar, radius_e8_Star/1e8)
        delta_hbarStar_all = np.append(delta_hbarStar_all, delta_hbar_star )
        all_radius[i] = radius_e8_a / 1e8
        all_delta_hbar[i] = delta_hbar_a

        np.save(f'./saved_data/matrice_star_{n}.npy', [radiusStar, delta_hbarStar_all])
        np.save(f'./saved_data/matrice_{n}.npy', np.array([all_radius], dtype=object))
        np.save(f'./saved_data/matrice_delta_hbar_{n}.npy', np.array([all_delta_hbar],dtype=object))


    all_radius_list = list(np.load(f'./saved_data/matrice_{n}.npy', allow_pickle=True))
    all_star_data = list(np.load(f'./saved_data/matrice_star_{n}.npy', allow_pickle=True))
    all_delta_hbar_list = list(np.load(f'./saved_data/matrice_delta_hbar_{n}.npy', allow_pickle=True))

    radius_star = all_star_data[0]
    delta_hbar_star = all_star_data[1]
    radius_a = all_radius_list[0]
    delta_hbar_data_a = all_delta_hbar_list[0]

    densities = densities / 1e12
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.gray_r
    adjusted_cmap = mcolors.LinearSegmentedColormap.from_list(
        'adjusted_gray_r', cmap(np.linspace(0.2, 0.8, 300)))
    colors = adjusted_cmap(densities)
    plt.plot(radius_star, delta_hbar_star,color='red', linestyle='--', label='WD surface', zorder=2)
    for i in range(len(radius_a)):
        ax.plot(radius_a[i], delta_hbar_data_a[i], color=colors[i], zorder = 1)
    norm = mcolors.Normalize(vmin=np.min(densities), vmax=np.max(densities))
    sm = plt.cm.ScalarMappable(cmap=adjusted_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=np.linspace(np.min(densities), np.max(densities), num=5))
    cbar.set_label('Core density ($10^{12}$kg/m$^3$)', fontsize=20)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    ax.set_xlabel('Distance (10$^5$km) ', fontsize=19)
    ax.set_ylabel(r'$\delta \hbar/\hbar_{\infty} $', fontsize=19)
    plt.ylim([5e-12, 4e-5])
    ax.set_yscale('log')
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.legend(loc = 'upper right')
    plt.savefig('./deltahbar_vs_radius_WD')
    # plt.show()


































