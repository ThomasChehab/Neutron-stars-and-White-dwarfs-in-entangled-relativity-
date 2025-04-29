#!/usr/bin/env python
import scipy.constants as cst
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from numpy import linalg as npla
from scipy.integrate import solve_ivp
from scipy.integrate import cumulative_trapezoid as integcum
from scipy.integrate import trapezoid as integ
import os
# import mplhep as hep
# hep.style.use("ATLAS")
import csv

#constants
c2 = cst.c**2
kappa = 8*np.pi*cst.G/c2**2
k = 1.475*10**(-3)*(cst.fermi**3/(cst.eV*10**6))**(2/3)*c2**(5/3)
massSun = 1.989*10**30

#Equation of state
def PEQS(Phi, rho, retro, dependence): # With or without retroaction
    if retro == False:
        return k*rho**(5/3)
    else:
        return k *rho**(5/3) * Phi**((-1*dependence)/2)

#Inverted equation of state
def RhoEQS(Phi, P, retro, dependence): # With or without retroaction
    if retro == False:
        return (P/k)**(3/5)
    else:
        return (P/k)**(3/5) * Phi**((3 * dependence)/10)


#Speed of sound (New)
# def v_sound_c(rho):
def v_sound_c(Phi, P, retro, dependence):
    if retro == False:
        return np.sqrt(5/3 * k * RhoEQS(Phi, P, retro, dependence)**(2/3)) / cst.c
    else:
        return np.sqrt(5/3 * Phi**((-1*dependence)/2) * k * RhoEQS(Phi, P, retro, dependence)**(2/3)) / cst.c

#Lagrangian
def Lagrangian(Phi, P, option, retro, dependence):
    rho = RhoEQS(Phi, P, retro, dependence)
    if option == 0:
        return -c2*rho+3*P
    elif option == 1:
        return -c2*rho
    elif option == 2:
        return P
    else:
        print('not a valid option')

#Equation for b
def b(r, m):
    return (1-(c2*m*kappa/(4*np.pi*r)))**(-1)

#Equation for da/dr
def adota(r, P, m, Psi, Phi):
    A = (b(r, m)/r)
    B = (1-(1/b(r, m))+P*kappa*r**2*Phi**(-1/2)-2*r*Psi/(b(r,m)*Phi))
    C = (1+r*Psi/(2*Phi))**(-1)
    return A*B*C

#Equation for D00
def D00(r, P, m, Psi, Phi, option, retro, dependence):
    ADOTA = adota(r, P, m, Psi, Phi)
    rho = RhoEQS(Phi, P, retro, dependence)
    Lm = Lagrangian(Phi, P, option, retro, dependence)
    T = -c2*rho + 3*P
    A = Psi*ADOTA/(2*Phi*b(r,m))
    B = kappa*(Lm-T)/(3*Phi**(1/2))
    return A+B

#Equation for db/dr
def bdotb(r, P, m, Psi, Phi, option, retro, dependence):
    rho = RhoEQS(Phi, P, retro, dependence)
    A = -b(r,m)/r
    B = 1/r
    C = b(r,m)*r*(-D00(r, P, m, Psi, Phi, option, retro, dependence)+kappa*c2*rho*Phi**(-1/2))
    return A+B+C

#Equation for dP/dr
def f1(r, P, m, Psi, Phi, option, retro, dependence):
    ADOTA = adota(r, P, m, Psi, Phi)
    Lm = Lagrangian(Phi, P, option, retro , dependence)
    rho = RhoEQS(Phi, P, retro, dependence)
    return -(ADOTA/2)*(P+rho*c2)+(Psi/(2*Phi))*(Lm-P)

#Equation for dm/dr
def f2(r, P, m, Psi, Phi, option, retro, dependence):
    rho = RhoEQS(Phi,P, retro, dependence)
    A = 4*np.pi*rho*(Phi**(-1/2))*r**2
    B = 4*np.pi*(-D00(r, P, m, Psi, Phi, option, retro, dependence)/(kappa*c2))*r**2
    return A+B

#Equation for dPhi/dr
def f3(r, P, m, Psi, Phi, option, dilaton_active):
    if dilaton_active:
        return Psi
    else:
        return 0

#Equation for dPsi/dr
def f4(r, P, m, Psi, Phi, option, dilaton_active, retro, dependence):
    ADOTA = adota(r, P, m, Psi, Phi)
    BDOTB = bdotb(r, P, m, Psi, Phi, option, retro, dependence)
    rho = RhoEQS(Phi,P, retro, dependence)
    Lm = Lagrangian(Phi, P, option, retro, dependence)
    T = -c2*rho + 3*P
    A = (-Psi/2)*(ADOTA-BDOTB+4/r)
    B = b(r,m)*kappa*Phi**(1/2)*(T-Lm)/3
    if dilaton_active:
        return A+B
    else:
        return 0

#Define for dy/dr
def dy_dr(r, y, option, dilaton_active, retro, dependence):
    P, M, Phi, Psi = y
    dy_dt = [f1(r, P, M, Psi, Phi,option, retro, dependence), f2(r, P, M, Psi, Phi, option, retro, dependence),f3(r, P, M, Psi, Phi, option, dilaton_active),f4(r, P, M, Psi, Phi, option, dilaton_active, retro, dependence)]
    return dy_dt

#Define for dy/dr out of the star
def dy_dr_out(r, y, P, option, dilaton_active, retro, dependence):
    M, Phi, Psi = y
    dy_dt = [f2(r, P, M, Psi, Phi, option, retro, dependence),f3(r, P, M, Psi, Phi, option, dilaton_active),f4(r, P, M, Psi, Phi, option, dilaton_active, retro, dependence)]
    return dy_dt

class TOV():
    """
    * Initialization
        - initDensity : initial value of density [MeV/fm3] (at the center of the star)
        - initPsi : initial value of psi (= 1).
        - initPhi : initial value for the derivative of psi (= 0).
        - radiusMax_in : For star interior, the solver integrates until it reach radiusMax_in.
        - radiusMax_out : For star exterior, the solver integrates until it reach radiusMax_out.
        - Npoint : Number at which the solution is evaluated (t_span Parameter in solve_ivp).
        - option : Select lagrangian.
            0 -> Lm=T
            1 -> Lm=-c²rho
            2 -> Lm=P
        - dilaton_active:
            True -> Solves for equation of ER.
            False -> Solves for equation of GR.
        - log_active: Consol outputs.
            True -> activates consol output
    """

    def __init__(self, initDensity, initPsi, initPhi, radiusMax_in, radiusMax_out, Npoint, EQS_type, dilaton_active, log_active, retro, dependence):
#Init value
        self.initDensity = initDensity
        self.initPsi = initPsi
        self.initPhi = initPhi
        self.initMass = 0
        self.initPressure = PEQS(initPhi, initDensity, retro, dependence)
#Parameters
        self.option = EQS_type
        self.dilaton_active = dilaton_active
        self.log_active = log_active
        self.retro = retro
#Computation variable
        self.radiusMax_in = radiusMax_in
        self.radiusMax_out = radiusMax_out
        self.Npoint = Npoint
        self.dependence = dependence
#Star data
        self.Nstar = 0
        self.massStar = 0
        self.massADM = 0
        self.pressureStar = 0
        self.radiusStar = 0
        self.phiStar = 0
#Output data
        self.pressure = 0
        self.mass = 0
        self.Phi = 0
        self.Psi = 0
        self.radius = 0
        self.g_tt = 0
        self.g_rr = 0
        self.g_tt_ext = 0
        self.g_rr_ext = 0
        self.r_ext = 0
        self.phi_inf = 0
        self.hbar = 0
        self.hbar_star = 0
        self.v_c = 0 # New

    def Compute(self):
        if self.log_active:
            print('===========================================================')
            print('SOLVER INSIDE THE STAR')
            print('===========================================================\n')
            print('Initial density: ', self.initDensity, ' MeV/fm^3')
            print('Initial pressure: ', self.initPressure/10**12, ' GPa')
            print('Initial mass: ', self.initMass/massSun, ' solar mass')
            print('Initial phi: ', self.initPhi)
            print('Initial psi: ', self.initPsi)
            print('Number of points: ', self.Npoint)
            print('Radius max: ', self.radiusMax_in/1000, ' km')
        y0 = [self.initPressure,self.initMass,self.initPhi,self.initPsi]
        if self.log_active:
            print('y0 = ', y0,'\n')
        r = np.linspace(0.01,self.radiusMax_in,self.Npoint)
        if self.log_active:
            print('radius min ',0.01)
            print('radius max ',self.radiusMax_in)
        sol = solve_ivp(dy_dr, [0.01, self.radiusMax_in], y0, method='RK45',t_eval=r ,dense_output = True, args=(self.option,self.dilaton_active, self.retro, self.dependence))
        # condition for Pressure = 0
        '''
        self.g_rr = b(sol.t, sol.y[1])
        a_dot_a = adota(sol.t, sol.y[0], sol.y[1], sol.y[3], sol.y[2])
        self.g_tt = np.exp(np.concatenate([[0.0], integcum(a_dot_a,sol.t)])-integ(a_dot_a,sol.t))
        plt.plot(self.g_tt/self.g_rr)
        plt.show()
        '''
        if sol.t[-1]<self.radiusMax_in:
            self.pressure = sol.y[0][0:-2]
            self.mass = sol.y[1][0:-2]
            self.Phi = sol.y[2][0:-2]
            self.v_c = v_sound_c(self.Phi, self.pressure, self.retro, self.dependence)
            self.Psi = sol.y[3][0:-2]
            self.radius = sol.t[0:-2]
            # Value at the radius of star
            self.massStar = sol.y[1][-1]
            self.radiusStar = sol.t[-1]
            self.pressureStar = sol.y[0][-1]
            self.phiStar = sol.y[2][-1]
            self.hbar_star = (self.phiStar)**(-1/2)
            n_star = len(self.radius)
            if self.log_active:
                print('Star radius: ', self.radiusStar/1000, ' km')
                print('Star Mass: ', self.massStar/massSun, ' solar mass')
                print('Star Mass: ', self.massStar, ' kg')
                print('Star pressure: ', self.pressureStar, ' Pa\n')
                print('===========================================================')
                print('SOLVER OUTSIDE THE STAR')
                print('===========================================================\n')
            y0 = [self.massStar, sol.y[2][-1],sol.y[3][-1]]
            if self.log_active:
                print('y0 = ', y0,'\n')
            r = np.logspace(np.log(self.radiusStar)/np.log(10),np.log(self.radiusMax_out)/np.log(10),self.Npoint)
            if self.log_active:
                print('radius min ',self.radiusStar)
                print('radius max ',self.radiusMax_out)
            sol = solve_ivp(dy_dr_out, [r[0], self.radiusMax_out], y0,method='DOP853', t_eval=r,max_step = 100000, args=(0,self.option,self.dilaton_active, self.retro, self.dependence))
            self.pressure = np.concatenate([self.pressure, np.zeros(self.Npoint)])
            self.mass = np.concatenate([self.mass, sol.y[0]])
            self.Phi = np.concatenate([self.Phi, sol.y[1]])
            self.Psi = np.concatenate([self.Psi,  sol.y[2]])
            self.radius = np.concatenate([self.radius, r])
            self.phi_inf = self.Phi[-1]
            self.hbar = 1/np.sqrt(self.Phi)
            if self.log_active:
                print('Phi at infinity ', self.phi_inf)
            # Compute metrics
            self.g_rr = b(self.radius, self.mass)
            a_dot_a = adota(self.radius, self.pressure, self.mass, self.Psi, self.Phi)
            #plt.plot(self.radius, np.concatenate([[0.0], integcum(a_dot_a,self.radius)]))
            #plt.show()
            self.g_tt = np.exp(np.concatenate([[0.0], integcum(a_dot_a,self.radius)])-integ(a_dot_a,self.radius))
            self.massADM = self.mass[-1]
            self.g_tt_ext = np.array(self.g_tt[n_star:-1])
            self.g_rr_ext = np.array(self.g_rr[n_star:-1])
            self.r_ext = np.array(self.radius[n_star:-1])
            self.r_ext[0] = self.radiusStar
            star_radius_normal = self.radiusStar
            if self.log_active:
                print('Star Mass ADM: ', self.massADM, ' kg')
                print('===========================================================')
                print('END')
                print('===========================================================\n')
        else:
            print('Pressure=0 not reached')

    def ComputeTOV(self):
        """
        ComputeTOV is the function to consider in order to compute "physical" quantities. It takes into account phi_inf->1 r->ininity
        """
        self.Compute()


    def ComputeTOV_normalization(self):
        self.Compute()
        if self.dilaton_active == True:
            self.initPhi /= self.phi_inf
            self.Compute()

    def find_dilaton_center(self):
        initDensity = self.initDensity
        dependence = self.dependence
        option = self.option
        precision = 1e-5#8
        retro = self.retro
        log_active = self.log_active
        dilaton_active = self.dilaton_active
        EQS_type = self.option
        radiusMax_out = self.radiusMax_out
        radiusMax_in = self.radiusMax_in
        Npoint = self.Npoint
        initPsi = 0
        radiusInit = 0.000001
        dilaton = True
        #Find limits of potential Phi_0
        Phi0_min, Phi0_max = 0.5, 1.5 # initial limits
        tov_min = TOV(initDensity, initPsi, Phi0_min, radiusMax_in, radiusMax_out, Npoint, EQS_type, dilaton_active, log_active, retro, dependence)
        tov_min.Compute()
        Phi_inf_min = tov_min.Phi[-1]
        while Phi_inf_min > 1:
            Phi0_min -= 0.1
            if Phi0_min == 0:
                Phi0_min = 1e-2
                 #print(f'Had to put l.h.s. limit of $\Phi_0$ to {Phi0_min}')
            tov_min = TOV(initDensity, initPsi, Phi0_min, radiusMax_in, radiusMax_out, Npoint, EQS_type, dilaton_active, log_active, retro, dependence)
            tov_min.Compute()
            Phi_inf_min = tov_min.Phi[-1]
             #print(f'Had to lower down the l.h.s.limit of $\Phi_0$ to {Phi0_min:.1f}')
        tov_max = TOV(initDensity, initPsi, Phi0_max, radiusMax_in, radiusMax_out, Npoint, EQS_type, dilaton_active, log_active, retro, dependence)
        tov_max.Compute()
        Phi_inf_max = tov_max.Phi[-1]
        while Phi_inf_max <1:
            Phi0_max += 0.1
            tov_max = TOV(initDensity, initPsi, Phi0_max, radiusMax_in, radiusMax_out, Npoint, EQS_type, dilaton_active, log_active, retro, dependence)
            tov_max.Compute()
            Phi_inf_max = tov_max.Phi[-1]
             #print(f'Had to increase the r.h.s. limit of $\Phi_0$ to {Phi0_max:.1f}')
        #Search for Phi_0 that leads to Phi_inf = 1 to a given precision by dichotomy
        step_precision = 1
        Phi0_dicho = np.array([Phi0_min, (Phi0_min + Phi0_max) / 2, Phi0_max])
        Phi_inf_dicho = np.zeros(3)
        while step_precision > precision:
            for n in range(3):
                tov = TOV(initDensity, initPsi, Phi0_dicho[n], radiusMax_in, radiusMax_out, Npoint, EQS_type, dilaton_active, log_active, retro, dependence)
                tov.Compute()
                Phi_inf_dicho[n] = tov.Phi[-1]
            N = np.min(np.argwhere(Phi_inf_dicho>1))
            Phi0_min = Phi0_dicho[N-1]
            Phi0_max = Phi0_dicho[N]
            Phi0_dicho = [Phi0_min, (Phi0_min + Phi0_max) / 2, Phi0_max]
            step_precision = np.abs(Phi_inf_dicho[N] - Phi_inf_dicho[N-1])
            Phi = (Phi0_min + Phi0_max) / 2
        return Phi, (Phi0_min + Phi0_max) / 2, (Phi0_min - Phi0_max) / 2, (Phi_inf_dicho[N] + Phi_inf_dicho[N-1]) / 2

def save_var_latex(key, value):
    dict_var = {}
    file_path = os.path.join(os.getcwd(), "NS_data.dat")
    dict_var[key] = value
    with open(file_path, "a") as f:
        for key in dict_var.keys():
            f.write(f"{key},{dict_var[key]}\n")

def save_var_latex_dependence(key, value):
    dict_var = {}
    file_path = os.path.join(os.getcwd(), "NS_hbar_dependency_data.dat")
    dict_var[key] = value
    with open(file_path, "a") as f:
        for key in dict_var.keys():
            f.write(f"{key},{dict_var[key]}\n")

def verify():
    if os.path.exists('./NS_data.dat'):
        os.remove('./NS_data.dat')
    if os.path.exists('./NS_hbar_dependency_data.dat'):
        os.remove('./NS_hbar_dependency_data.dat')

def verify_files(matrices):
    missing_file = [files for files in matrices if not os.path.exists(files)]

    if missing_file:
        print("Following files are missing :\n")
        for fichier in missing_file:
            print(f"- {fichier}")
        return False  # if one files is missing
    else:
        print("All files are present.\n")
        return True  # Each files exists

