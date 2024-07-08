#!/usr/bin/env python
import scipy.constants as cst
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as npla
import os
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
import mplhep as hep
hep.style.use("ATLAS")

#constants
c2 = cst.c**2
kappa = 8*np.pi*cst.G/c2**2
k = 1.475*10**(-3)*(cst.fermi**3/(cst.eV*10**6))**(2/3)*c2**(5/3)
massSun = 1.989*10**30

#Equation of state
def PEQS(Phi, rho, retro):
    if retro == False: # Without or With retroaction
        return k*rho**(5/3) 
    else:
        return k*rho**(5/3) * Phi**(-1)

#Inverted equation of state
def RhoEQS(Phi, P, retro):
    if retro == False: # Without or With retroaction
        return (P/k)**(3/5)
    else:
        return (P/k)**(3/5) * Phi**(3/5)

#Lagrangian
def Lagrangian(Phi, P, option, retro):
    rho = RhoEQS(Phi, P, retro)
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
def D00(r, P, m, Psi, Phi, option, retro):
    ADOTA = adota(r, P, m, Psi, Phi)
    rho = RhoEQS(Phi, P, retro)
    Lm = Lagrangian(Phi, P, option, retro)
    T = -c2*rho + 3*P
    A = Psi*ADOTA/(2*Phi*b(r,m))
    B = kappa*(Lm-T)/(3*Phi**(1/2))
    return A+B

#Equation for db/dr
def bdotb(r, P, m, Psi, Phi, option, retro):
    rho = RhoEQS(Phi, P, retro)
    A = -b(r,m)/r
    B = 1/r
    C = b(r,m)*r*(-D00(r, P, m, Psi, Phi, option, retro)+kappa*c2*rho*Phi**(-1/2))
    return A+B+C

#Equation for dP/dr
def f1(r, P, m, Psi, Phi, option, retro):
    ADOTA = adota(r, P, m, Psi, Phi)
    Lm = Lagrangian(Phi, P, option, retro)
    rho = RhoEQS(Phi, P, retro)
    return -(ADOTA/2)*(P+rho*c2)+(Psi/(2*Phi))*(Lm-P)

#Equation for dm/dr
def f2(r, P, m, Psi, Phi, option, retro):
    rho = RhoEQS(Phi,P, retro)
    A = 4*np.pi*rho*(Phi**(-1/2))*r**2
    B = 4*np.pi*(-D00(r, P, m, Psi, Phi, option, retro)/(kappa*c2))*r**2
    return A+B

#Equation for dPhi/dr
def f3(r, P, m, Psi, Phi, option, retro):
    ADOTA = adota(r, P, m, Psi, Phi)
    BDOTB = bdotb(r, P, m, Psi, Phi, option, retro)
    rho = RhoEQS(Phi,P, retro)
    Lm = Lagrangian(Phi, P, option, retro)
    T = -c2*rho + 3*P
    A = (-Psi/2)*(ADOTA-BDOTB+4/r)
    B = b(r,m)*kappa*Phi**(1/2)*(T-Lm)/3
    return A+B

#Equation for dPsi/dr
def f4(r, P, m, Psi, Phi, option):
    return Psi

class TOV():

    def __init__(self, initRadius, initDensity, initPsi, PhiInit, radiusStep, option, dilaton, precision, retro):
#Init value
        self.initDensity = initDensity
        self.initPressure = PEQS(PhiInit, initDensity, retro)
        self.initPsi = initPsi
        self.PhiInit = PhiInit
        self.initMass = 0
        self.initRadius = initRadius
#Computation variable
        self.limitCompute = 10000
        self.radiusStep = radiusStep
#Star data
        self.massStar = 0
        self.pressureStar = 0
        self.radiusStar = 0
        self.stepStar = 0
        self.LambdaStar = 0
#Output vector
        self.pressure = 0
        self.mass = 0
        self.Phi = 0
        self.Psi = 0
        self.metric00 = 0
        self.metric11 = 0
        self.radius = 0
        self.PhiInf = 0
        self.hbar = 0
        self.hbar_retro = 0
        self.option = option
        self.dilaton = dilaton
        self.precision = precision
        self.retro = retro

    def Compute(self):
# Initialisation ===================================================================================
        dr = self.radiusStep
        n = 0
        P =        [self.initPressure]
        m =        [self.initMass]
        Psi =      [self.initPsi]
        Phi =      [self.PhiInit]
        r =        [self.initRadius]
        metric11 = [b(r[0],m[0])]
        metric00 = [1]      # Coordinate time as proper time at center
        # Inside the star----------------------------------------------------------------------------
        while(P[n]>10**26):
            if(n == self.limitCompute):
                break
                print('end')
            else:
                P.append(     P[n]   + dr*f1(r[n], P[n], m[n], Psi[n], Phi[n], self.option, self.retro ))
                m.append(     m[n]   + dr*f2(r[n], P[n], m[n], Psi[n], Phi[n], self.option, self.retro ))
                if self.dilaton:
                    Phi.append(   Phi[n] + dr*f4(r[n], P[n], m[n], Psi[n], Phi[n], self.option))
                    Psi.append(   Psi[n] + dr*f3(r[n], P[n], m[n], Psi[n], Phi[n], self.option, self.retro ))
                else:
                    Phi.append(   Phi[n] )
                    Psi.append(   Psi[n] )
                metric11.append(b(r[n], m[n]))
                metric00.append(metric00[n] + dr*metric00[n]*adota(r[n], P[n], m[n], Psi[n], Phi[n]))
                n = n+1
                r.append(self.initRadius+n*dr)
        P.pop()
        m.pop()
        Psi.pop()
        Phi.pop()
        r.pop()
        metric11.pop()
        metric00.pop()
        n = n-1
        self.pressure = P #star pressure
        self.mass = m #star mass
        self.stepStar = n #index where code stop at 0 pressure


        # Outside the star--------------------------------------------------------------------------
        while(n<self.limitCompute):
            if self.dilaton:
                Phi.append( Phi[n] + dr*f4(r[n], 0, m[n], Psi[n], Phi[n], self.option ))
                Psi.append( Psi[n] + dr*f3(r[n], 0, m[n], Psi[n], Phi[n], self.option, self.retro ))
            else:
                Phi.append( Phi[n] )
                Psi.append( Psi[n] )
            P.append( P[n] + dr*f1(r[n], P[n], m[n], Psi[n], Phi[n], self.option, self.retro ))
            m.append( m[n] + dr*f2(r[n], 0, m[n], Psi[n], Phi[n], self.option, self.retro ))
            metric11.append(b(r[n], m[n]))
            metric00.append(metric00[n] + dr*metric00[n]*adota(r[n],0, m[n], Psi[n], Phi[n]))
            n = n+1
            r.append(self.initRadius+n*dr)

        # Star property
        self.massStar = self.mass[self.stepStar]
        self.radiusStar = r[self.stepStar]
        self.pressureStar = self.pressure[self.stepStar]
        self.Psi = Psi
        self.Phi = Phi
        self.radius = r
        self.metric11 = metric11
        self.metric00 = metric00
        self.hbar = 1/np.sqrt(self.Phi) #New
        hbar_core = np.sqrt(self.Phi[0])
        #print(' Star mass =', self.mass[self.stepStar]/massSun)
        #print('Star radius =', r[self.stepStar])
        #print('Star Phi =', self.Phi[self.stepStar])
        #print('Phi inf =', self.Phi[-1])
        #print('HBAR CORE = ', hbar_core) #New

    def ComputeGR(self):
        dr = self.radiusStep
        n = 0 #Integration parameter
        P =        [self.initPressure]
        m =        [self.initMass]
        r =        [self.initRadius]
        metric11 = [b(r[0],m[0])]
        metric00 = [1]      # Coordinate time as proper time at center
        while(P[n]>10**26):
            if(n == self.limitCompute):
                break
                print('end')
            else:
                P.append( P[n] + dr*f1(r[n], P[n], m[n], 0, 1, self.option, self.retro ))
                m.append( m[n] + dr*f2(r[n], P[n], m[n], 0, 1, self.option, self.retro ))
                metric11.append(b(r[n], m[n]))
                metric00.append(metric00[n] + dr*metric00[n]*adota(r[n],0, m[n], 0, 1))
                n = n+1
                r.append(self.initRadius+n*dr)
        P.pop()
        m.pop()
        r.pop()
        metric00.pop()
        metric11.pop()
        n = n-1
        self.massStar = m[n]
        self.radiusStar = r[n]
        self.pressureStar = P[n]
        self.pressure = P
        self.mass = m
        self.metric00 = metric00
        self.metric11 = metric11


    def PlotEvolution(self):
        plt.figure()
        plt.plot([x/10**3 for x in self.radius[0:self.stepStar*2]], [x for x in self.pressure[0:self.stepStar*2]])
        plt.xlabel('Radius r (km)')
        plt.title('Pressure P (Pa)', fontsize=12)

        plt.figure()
        plt.plot([x/10**3 for x in self.radius[0:self.stepStar*2]], [x/(1.989*10**30) for x in self.mass[0:self.stepStar*2]])
        plt.xlabel('Radius r (km)')
        plt.title('Mass $M/M_{\odot}$', fontsize=12)

        plt.figure()
        plt.plot([x/10**3 for x in self.radius[0:self.stepStar*20]], self.Phi[0:self.stepStar*20])
        plt.xlabel('Radius r (km)')
        plt.title('Dilaton field Φ', fontsize=12)

        plt.figure()
        plt.plot([x/10**3 for x in self.radius[0:self.stepStar*20]], self.Psi[0:self.stepStar*20])
        plt.xlabel('Radius r (km)')
        plt.title('Ψ (derivative of Φ)', fontsize=12)

        plt.figure()
        ax3 = plt.subplot(1,2,1)
        plt.plot([x/10**3 for x in self.radius], self.metric00)
        plt.xlabel('Radius r (km)')
        plt.ylabel('a')

        ax4 = plt.subplot(1,2,2)
        plt.plot([x/10**3 for x in self.radius], self.metric11)
        plt.xlabel('Radius r (km)')
        plt.ylabel('b')

        plt.show()

    def Phi_infini(self):
        return self.Phi[self.stepStar*10]

    def Phi_r(self):
        M = np.zeros(len(self.Phi))
        for i in range(len(self.Phi)):
            M[i] = self.Phi[i]-self.PhiInit
        plt.plot([x/10**3 for x in self.radius[:]], M)
        
        
        
        
#########################################################################################        
    
    def find_dilaton_center(self):
        
        initDensity = self.initDensity
        radiusStep = self.radiusStep
        option = self.option
        precision = self.precision
        retro = self.retro
        PsiInit = 0
        radiusInit = 0.000001
        dilaton = True
        
        #Find limits of potential Phi_0
        Phi0_min, Phi0_max = 0.5, 1.5 # initial limits
        
        tov_min = TOV(radiusInit, initDensity, PsiInit, Phi0_min, radiusStep, option, dilaton, precision, retro)
        tov_min.Compute()
        Phi_inf_min = tov_min.Phi[-1]
        while Phi_inf_min > 1:
            Phi0_min -= 0.1
            if Phi0_min == 0:
                Phi0_min = 1e-2
    #             print(f'Had to put l.h.s. limit of $\Phi_0$ to {Phi0_min}')
            tov_min = TOV(radiusInit, initDensity, PsiInit, Phi0_min, radiusStep, option, dilaton, precision, retro)
            tov_min.Compute()
            Phi_inf_min = tov_min.Phi[-1]
    #         print(f'Had to lower down the l.h.s.limit of $\Phi_0$ to {Phi0_min:.1f}')
            
        tov_max = TOV(radiusInit, initDensity, PsiInit, Phi0_max, radiusStep, option, dilaton, precision, retro)
        tov_max.Compute()
        Phi_inf_max = tov_max.Phi[-1]
        while Phi_inf_max <1:
            Phi0_max += 0.1
            tov_max = TOV(radiusInit, initDensity, PsiInit, Phi0_max, radiusStep, option, dilaton, precision, retro)
            tov_max.Compute()
            Phi_inf_max = tov_max.Phi[-1]
    #         print(f'Had to increase the r.h.s. limit of $\Phi_0$ to {Phi0_max:.1f}')
            
        #Search for Phi_0 that leads to Phi_inf = 1 to a given precision by dichotomy
        step_precision = 1
        Phi0_dicho = np.array([Phi0_min, (Phi0_min + Phi0_max) / 2, Phi0_max])
        Phi_inf_dicho = np.zeros(3)
        while step_precision > precision:
            for n in range(3):
                tov = TOV(radiusInit, initDensity, PsiInit, Phi0_dicho[n], radiusStep, option, dilaton, precision, retro)
                tov.Compute()
                Phi_inf_dicho[n] = tov.Phi[-1] 
            N = np.min(np.argwhere(Phi_inf_dicho>1))
            Phi0_min = Phi0_dicho[N-1]
            Phi0_max = Phi0_dicho[N]
            Phi0_dicho = [Phi0_min, (Phi0_min + Phi0_max) / 2, Phi0_max]
            step_precision = np.abs(Phi_inf_dicho[N] - Phi_inf_dicho[N-1])
            Phi = (Phi0_min + Phi0_max) / 2
            self.PhiInit = Phi
        return Phi, (Phi0_min + Phi0_max) / 2, (Phi0_min - Phi0_max) / 2, (Phi_inf_dicho[N] + Phi_inf_dicho[N-1]) / 2
            

############################################################################################
            
            
    #Recording hbar data in a specific folder
    def hbar_into_txt(self):
        folder_path = './hbar_folder'
        if not os.path.exists(folder_path): 
            os.makedirs(folder_path) 
        Name = "./hbar_folder/hbar_data.txt"
        if not os.path.exists(Name):
            open(Name, 'w').close()  
        with open(Name, 'w') as f:
            for element in self.hbar:
                f.write(str(element) + '\n')
                
    #Recording radius data in a specific folder
    def radius_into_txt(self):
        folder_path = './radius_folder'
        if not os.path.exists(folder_path): 
            os.makedirs(folder_path) 
        Name = "./radius_folder/radius_data.txt"
        if not os.path.exists(Name):
            open(Name, 'w').close()  
        with open(Name, 'w') as f:
            for element in self.radius:
                f.write(str(element) + '\n')

                
    #Recording radius_retro data in a specific folder
    def radius_retro_into_txt(self):
        folder_path = './radius_folder'
        if not os.path.exists(folder_path): 
            os.makedirs(folder_path) 
        Name = "./radius_folder/radius_retro_data.txt"
        if not os.path.exists(Name):
            open(Name, 'w').close()  
        with open(Name, 'w') as f:
            for element in self.radius:
                f.write(str(element) + '\n')


    #Recording hbar_retro data in a specific folder
    def hbar_retro_into_txt(self):
        folder_path = './hbar_folder'
        if not os.path.exists(folder_path): 
            os.makedirs(folder_path) 
        Name = "./hbar_folder/hbar_retro_data.txt"
        if not os.path.exists(Name):
            open(Name, 'w').close()  
        with open(Name, 'w') as f:
            for element in self.hbar:
                f.write(str(element) + '\n')

############################################################################""

    def Plot(self):
        #Recovering hbar data
        hbar = []
        file_path = ('./hbar_folder/hbar_data.txt')
        f = open(file_path, 'r')
        for x in f:
            hbar.append(x)
        for i in range(len(hbar)):
            hbar[i] = float(hbar[i])
                
                
        #Recovering radius data
        radius = []
        file_path = ('./radius_folder/radius_data.txt')
        f = open(file_path, 'r')
        for x in f:
            radius.append(x)
        for i in range(len(radius)):
            radius[i] = float(radius[i])
            
        #Recovering hbar_retro data
        hbar_retro = []
        file_path = ('./hbar_folder/hbar_retro_data.txt')
        f = open(file_path, 'r')
        for x in f:
            hbar_retro.append(x)
        for i in range(len(hbar_retro)):
            hbar_retro[i] = float(hbar_retro[i])
                
        #Recovering radius_retro data
        radius_retro = []
        file_path = ('./radius_folder/radius_retro_data.txt')
        f = open(file_path, 'r')
        for x in f:
            radius_retro.append(x)
        for i in range(len(radius_retro)):
            radius_retro[i] = float(radius_retro[i])
            
        radius_normal = np.array(radius)
        radius_normal /= 1e3
        radius_retro = np.array(radius_retro)
        radius_retro /= 1e3
        hbar_normal = np.array(hbar)
        hbar_retro = np.array(hbar_retro)
        
        #Plot
        star_radius= self.radiusStar/1e3
        plt.figure()
        plt.plot(radius_retro, hbar_retro, label = 'With retroaction')
        plt.plot(radius_normal, hbar_normal, label = 'Without retroaction' )
        plt.xlim(-2, 60)
        plt.ylim(0.999,1.045)
        plt.axvline(star_radius, color='r', linestyle='--', label='Star radius')
        plt.fill_between(radius_retro, hbar_normal, hbar_retro, where=(hbar_retro > hbar_normal), color='lightgray', alpha=0.5)
        plt.xlabel('Radius (km) $\\times$ 1e3', fontsize=19)
        plt.ylabel(r'$\hbar$ variation', fontsize=22)
        plt.legend()
        plt.savefig('./hbar_variation_comparison_NS')
        #plt.show()



                
                
                
                
                
                
                
                
                
                
                
