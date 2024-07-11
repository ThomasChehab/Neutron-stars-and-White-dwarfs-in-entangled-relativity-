#!/usr/bin/env python
import scipy.constants as cst
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as npla
from scipy.integrate import solve_ivp
from scipy.integrate import cumtrapz as integcum
from scipy.integrate import trapz as integ
import os
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
import mplhep as hep
hep.style.use("ATLAS")

#Constants
c2 = cst.c**2
kappa = 8*np.pi*cst.G/c2**2
massSun = 1.989*10**30
A = 6.02e+21 # cst
B = 9.82e8 * (4.002602/2) # cst * molecular weight of helium

#Equation of state
def PEQS(x):
    f = ((x*(2*x**2 - 3 )*(1+x**2)**(1/2)) + 3 * np.arcsinh(x)) 
    return A * f

def RhoEQS(x):
    return B * x**3

#Inverted equation of state
def xEQS(rho):
    return (rho/B)**(1/3)

#Lagrangian
def Lagrangian(P,x, option):
    rho = RhoEQS(x)
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
def D00(r, P, m, Psi, Phi,x, option):
    ADOTA = adota(r, P, m, Psi, Phi)
    Lm = Lagrangian(P,x, option)
    rho = RhoEQS(x)
    T = -c2*rho + 3*P
    A = Psi*ADOTA/(2*Phi*b(r,m))
    B = kappa*(Lm-T)/(3*Phi**(1/2))
    return A+B

#Equation for db/dr
def bdotb(r, P, m, Psi, Phi,x, option):
    rho = RhoEQS(x)
    A = -b(r,m)/r
    B = 1/r
    C = b(r,m)*r*(-D00(r, P, m, Psi, Phi,x, option)+kappa*c2*rho*Phi**(-1/2))
    return A+B+C

#Equation for dP/dr
def f1(r, P, m, Psi, Phi,x, option):
    ADOTA = adota(r, P, m, Psi, Phi)
    Lm = Lagrangian(P,x, option)
    rho = RhoEQS(x)
    return -(ADOTA/2)*(P+rho*c2)+(Psi/(2*Phi))*(Lm-P)

#Equation for dm/dr
def f2(r, P, m, Psi, Phi, x, option):
    rho = RhoEQS(x)
    A = 4*np.pi*rho*(Phi**(-1/2))*r**2
    B = 4*np.pi*(-D00(r, P, m, Psi, Phi,x, option)/(kappa*c2))*r**2
    return A+B

#Equation for dPsi/dr
def f4(r, P, m, Psi, Phi,x, option, dilaton_active):
    ADOTA = adota(r, P, m, Psi, Phi)
    BDOTB = bdotb(r, P, m, Psi, Phi, x, option)
    rho = RhoEQS(x)
    Lm = Lagrangian(P,x, option)
    T = -c2*rho + 3*P
    A = (-Psi/2)*(ADOTA-BDOTB+4/r)
    B = b(r,m)*kappa*Phi**(1/2)*(T-Lm)/3
    if dilaton_active:
        return A+B 
    else:
        return 0

#Equation for dPhi/dr
def f3(r, P, m, Psi, Phi, option, dilaton_active):
    if dilaton_active:
        return Psi
    else:
        return 0
    
#Equation for dx/dr
def fx(r, P, m, Psi, Phi,x, option):
    rho = RhoEQS(x)
    ADOTA = adota(r, P, m, Psi, Phi)
    Lm = Lagrangian(P, x, option)
    return (-(ADOTA/2)*(P+rho*c2)+(Psi/(2*Phi))*(Lm-P))*(((x**2+1)**(1/2))/(A*8*x**4))

#Define for dy/dr
def dy_dr(r, y, option, dilaton_active):
    P, M, Phi, Psi, x = y
    dy_dt = [f1(r, P, M, Psi, Phi,x,option), f2(r, P, M, Psi, Phi,x, option),f3(r, P, M, Psi, Phi, option, dilaton_active),f4(r, P, M, Psi, Phi,x, option, dilaton_active),fx(r, P, M, Psi, Phi,x, option) ]    
    return dy_dt

#Define for dy/dr out of the star
def dy_dr_out(r, y, P, x, option, dilaton_active):
    M, Phi, Psi = y
    dy_dt = [f2(r, P, M, Psi, Phi,x, option),f3(r, P, M, Psi, Phi, option, dilaton_active),f4(r, P, M, Psi, Phi,x, option, dilaton_active)]
    return dy_dt

class TOV():

    def __init__(self, initDensity, initPsi, initPhi, radiusMax_in, radiusMax_out, Npoint, EQS_type, dilaton_active, log_active):
        
#Init value
        self.initx = xEQS(initDensity)
        self.initDensity = initDensity
        self.initPressure = PEQS(self.initx)
        self.initPsi = initPsi
        self.initPhi = initPhi
        self.initMass = 0
        self.option = EQS_type
        self.dilaton_active = dilaton_active
        self.log_active = log_active

#Computation variable
        self.radiusMax_in = radiusMax_in
        self.radiusMax_out = radiusMax_out
        self.Npoint = Npoint
#Star data
        self.Nstar = 0
        self.massStar = 0
        self.massADM = 0
        self.pressureStar = 0
        self.radiusStar = 0
        self.phiStar = 0
        self.hbarStar = 0 #New
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
        self.hbar = 0 #New
        self.hbar_inf = 0 #New
        self.delta_hbar = 0 #New

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
        y0 = [self.initPressure, self.initMass,self.initPhi,self.initPsi,self.initx]
        if self.log_active:
            print('y0 = ', y0,'\n')
        r = np.linspace(0.01,self.radiusMax_in,self.Npoint) 
        if self.log_active:
            print('radius min ',0.01)
            print('radius max ',self.radiusMax_in)
        
        def event(t, y, option, dilaton_active): #New #If the sign of x changes, integration stops
            return np.sign(1) == np.sign(y[4])
        event.terminal = True
        event.direction = 0
        
        sol = solve_ivp(dy_dr, [0.01, self.radiusMax_in], y0, method='RK45',t_eval=r, events = event, args=(self.option,self.dilaton_active)) 
        
        # condition for Pressure = 0
        '''
        self.g_rr = b(sol.t, sol.y[1])
        a_dot_a = adota(sol.t, sol.y[0], sol.y[1], sol.y[3], sol.y[2])
        self.g_tt = np.exp(np.concatenate([[0.0], integcum(a_dot_a,sol.t)])-integ(a_dot_a,sol.t))
        plt.plot(self.g_tt/self.g_rr)
        plt.show()
        '''

        if sol.t[-1]<self.radiusMax_in:    
            
            self.radius = sol.t[0:-2]
            self.x = sol.y[4][0:-2]
            self.xStar = self.x[-1]
            self.pressure = sol.y[0][0:-2]
            self.mass = sol.y[1][0:-2]
            self.Phi = sol.y[2][0:-2]
            self.Psi = sol.y[3][0:-2]

            # Value at the radius of star
            self.massStar = sol.y[1][-1]
            self.radiusStar = sol.t[-1]
            self.pressureStar = sol.y[0][-1]
            self.phiStar = sol.y[2][-1]
            self.xStar = sol.y[4][-1]
            n_star = len(self.radius)
                      
            if self.log_active:
                print('Star radius: ', self.radiusStar/1000, ' km')
                print('Star Mass: ', self.massStar/massSun, ' solar mass')
                print('Star Mass: ', self.massStar, ' kg')
                print('Star pressure: ', self.pressureStar, ' Pa\n')
                print('Star Phi: ', self.phiStar)
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
                
            sol = solve_ivp(dy_dr_out, [r[0], self.radiusMax_out], y0,method='DOP853', t_eval=r,max_step = 100000, args=(0,0,self.option,self.dilaton_active)) #New #Steps needs to not be too high to obtain smooth results
            
            self.pressure = np.concatenate([self.pressure, np.zeros(self.Npoint)])
            self.x = np.concatenate([self.x, np.zeros(self.Npoint)])
            self.mass = np.concatenate([self.mass, sol.y[0]])
            self.Phi = np.concatenate([self.Phi, sol.y[1]])
            self.Psi = np.concatenate([self.Psi,  sol.y[2]])
            self.radius = np.concatenate([self.radius, r])
            self.phi_inf = self.Phi[-1]
            if self.log_active:
                print('Phi at infinity ', self.phi_inf)
            # Compute metrics
            self.g_rr = b(self.radius, self.mass)
            a_dot_a = adota(self.radius, self.pressure, self.mass, self.Psi, self.Phi)
            self.g_tt = np.exp(np.concatenate([[0.0], integcum(a_dot_a,self.radius)])-integ(a_dot_a,self.radius))
            self.massADM = self.mass[-1]
            self.g_tt_ext = np.array(self.g_tt[n_star:-1])
            self.g_rr_ext = np.array(self.g_rr[n_star:-1])
            self.r_ext = np.array(self.radius[n_star:-1])
            self.r_ext[0] = self.radiusStar
            
            #Compute hbar variation
            self.hbar = 1/np.sqrt(self.Phi)
            self.hbar_inf = 1/np.sqrt(self.phi_inf)
            self.hbarStar = 1/np.sqrt(self.phiStar)
            self.delta_hbar = (self.hbar - self.hbar_inf)/self.hbar_inf
            
            if self.log_active:
                print('Star Mass ADM: ', self.massADM, ' kg')
                print('hbar variation in % =', self.delta_hbar * 100)
                print('===========================================================')
                print('END')
                print('===========================================================\n')
        else:
            print('Pressure=0 not reached')


    def ComputeTOV(self):
        """
        ComputeTOV is the function to consider in order to compute "physical" quantities. It takes into account phi_inf->1 r->infinity
        """
        self.Compute()
        if self.dilaton_active:
            self.initPhi = self.initPhi/self.phi_inf
            self.Compute()
            
    #Next functions are used to store in folder white dwarfs data
            
    def density_into_txt(self): # Storing density data
        folder_path = './init_density_folder'
        if not os.path.exists(folder_path): 
            os.makedirs(folder_path) 
        Init_density = []
        Init_density.append(self.initDensity)
        Name = "./init_density_folder/init_density.txt"
        if not os.path.exists(Name):
            open(Name, 'a').close()
        with open(Name, 'a') as f:
            for element in Init_density:
                f.write(str(element) + '\n')
            
    def hbar_into_txt(self,i): # Storing hbar variation data
        folder_path = './hbar_folder'
        if not os.path.exists(folder_path): 
            os.makedirs(folder_path) 
        Name = f"./hbar_folder/hbar_data{i}.txt"
        if not os.path.exists(Name):
            open(Name, 'w').close()  
        with open(Name, 'w') as f:
            for element in self.delta_hbar:
                f.write(str(element) + '\n')

    def radius_into_txt(self,i): # Storing radius data
        folder_path = './radius_folder'
        if not os.path.exists(folder_path): 
            os.makedirs(folder_path) 
        Name = f"./radius_folder/radius_data{i}.txt"
        if not os.path.exists(Name):
            open(Name, 'w').close()  
        with open(Name, 'w') as f:
            for element in self.radius:
                f.write(str(element) + '\n')
        
    #Next function goal is to recover data of white dwarf and obtain the final plot. 
    def Plot_all_hbar(self):
        
        #Recovering density data
        density = []
        file_path_density = ('./init_density_folder/init_density.txt')
        f = open(file_path_density, 'r')
        for x in f:
            density.append(x)
        for i in range(len(density)):
            density[i] = float(density[i])
        density = np.array(density)/(1e12)
        
        #Recovering hbar data
        hbar = []
        for i in range(len(density)):
            file_path_hbar = (f'./hbar_folder/hbar_data{i}.txt')
            f = open(file_path_hbar, 'r')
            hbar_0 = []
            for x in f:
                hbar_0.append(x)
            hbar.append(hbar_0)
        for i in range(len(hbar)):
            for j in range(len(hbar[i])):
                hbar[i][j] = float(hbar[i][j])
                 
        #Recovering radius data
        radius = []
        for i in range(len(density)):
            file_path_radius = (f'./radius_folder/radius_data{i}.txt')
            f = open(file_path_radius, 'r')
            radius_0 = []
            for x in f:
                radius_0.append(x)
            radius.append(radius_0)

        for i in range(len(radius)):
            for j in range(len(radius[i])):
                radius[i][j] = float(radius[i][j])
                radius[i][j] /= 1e8 
                
        #Plot
        fig, ax = plt.subplots(figsize=(11, 6))
        cmap = plt.cm.gray_r
        adjusted_cmap = mcolors.LinearSegmentedColormap.from_list(
            'adjusted_gray_r', cmap(np.linspace(0.2, 0.8, 300)))
        colors = adjusted_cmap(density)
        for i in range(len(hbar)):
            ax.plot(radius[i], hbar[i], color=colors[i])
        norm = mcolors.Normalize(vmin=np.min(density), vmax=np.max(density))
        sm = plt.cm.ScalarMappable(cmap=adjusted_cmap, norm=norm)
        sm.set_array([]) 
        cbar = fig.colorbar(sm, ax=ax, ticks=np.linspace(np.min(density), np.max(density), num=5))
        cbar.set_label('Core density (kg/m$^3$) $\\times$ 1e12 ', fontsize=20)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        ax.set_xlabel('Radius (km) $\\times$ 1e5', fontsize=19)
        ax.set_ylabel(r'$\delta \hbar = \hbar - \hbar|_{r \rightarrow \infty}$', fontsize=19) 
        plt.ylim([5e-12, 4e-5])
        ax.set_yscale('log')
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        plt.savefig('./deltahbar_vs_radius_WD')
            
        
    def Plot(self):
        plt.subplot(221)
        plt.plot([x/10**3 for x in self.radius], [x for x in self.pressure])
        plt.xlabel('Radius r (km)')
        plt.title('Pressure P (Pa)', fontsize=12)
        plt.axvline(x=self.radiusStar/10**3, color='r')

        plt.subplot(222)
        plt.plot([x/10**3 for x in self.radius], [x/massSun for x in self.mass])
        plt.xlabel('Radius r (km)')
        plt.title('Mass $M/M_{\odot}$', fontsize=12)
        plt.axvline(x=self.radiusStar/10**3, color='r')

        plt.subplot(223)
        plt.plot([x/10**3 for x in self.radius], self.Phi)
        plt.xlabel('Radius r (km)')
        plt.title('Dilaton field Φ', fontsize=12)
        plt.axvline(x=self.radiusStar/10**3, color='r')

        plt.subplot(224)
        plt.plot([x/10**3 for x in self.radius], self.Psi)
        plt.xlabel('Radius r (km)')
        plt.title('Ψ (derivative of Φ)', fontsize=12)
        plt.axvline(x=self.radiusStar/10**3, color='r')

        plt.show()

    def PlotMetric(self):
        plt.subplot(121)
        plt.plot([x/10**3 for x in self.radius], self.g_tt)
        plt.xlabel('Radius r (km)')
        plt.title('g_tt', fontsize=12)
        plt.axvline(x=self.radiusStar/10**3, color='r')

        plt.subplot(122)
        plt.plot([x/10**3 for x in self.radius], self.g_rr)
        plt.xlabel('Radius r (km)')
        plt.title('g_rr', fontsize=12)
        plt.axvline(x=self.radiusStar/10**3, color='r')

        plt.show()
