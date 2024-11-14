import numpy as np
import os
import pandas as pd
import sympy as sp
import symengine as se
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
import math
from pathlib import Path
import time
from pyphasefield.field import Field
from pyphasefield.simulation import Simulation
from pyphasefield.ppf_utils import COLORMAP_OTHER, COLORMAP_PHASE, make_seed
import random
try:
    from numba import cuda
    import numba
    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
except:
    import pyphasefield.jit_placeholder as numba
    import pyphasefield.jit_placeholder as cuda
# pay attention to the BC init and the output when data transtfer from GPU, the random for interface has been set
#with moving frame
@numba.jit
def gibbsliq(X, T, R, VM):
    gibbsTiliq = -19887.066 + 298.7367 * T - 46.29 * T * math.log(T)
    gibbsNbliq = -8519.353 + 142.045475 * T - 26.4711 * T * math.log(T) + 0.203475E-3 * T ** 2 - 0.35012E-6 * T ** 3 \
                 + 93399 * T ** (-1) + 29781.555 - 10.816418 * T- 306.098E-25 * T ** 7
    LNbTi = 7406.1
    gibbs_liq = (X * gibbsNbliq + (1 - X) * gibbsTiliq + R * T * (X * math.log(X) + (1.0 - X) * math.log(1.0 - X)) + \
                 X * (1.0 - X) * LNbTi) / VM
    return gibbs_liq

@numba.jit
def gibbsFCC(X,T,R,VM):
    gibbs_Nb_FCC = -8519.353 + 142.045475 * T - 26.4711 * T * math.log(T) + 0.203475E-3 * T ** 2 - 0.35012E-6 * T ** 3 + \
                   93399 * T ** (-1)
    gibbs_Ti_FCC = 26483.26 - 182.426471 * T + 19.0900905 * T * math.log(T) - 22.00832E-3 * T ** 2 + 1.228863E-6 * T ** 3 + \
                       1400501 * T ** (-1)
    LNbTi = 13045.3
    gibbs_FCC = (X* gibbs_Nb_FCC + (1-X)*gibbs_Ti_FCC + R*T * (X*math.log(X) + (1-X)* math.log(1-X)) + X*(1-X) * LNbTi)/VM
    return gibbs_FCC

@numba.jit
def Dgibbsliq(X,T,R,VM):
    gibbsNbliq = -8519.353 + 142.045475 * T - 26.4711 * T * math.log(T) + 0.203475e-3 * T**2 - 0.35012e-6 * T**3 + \
                    93399 * T**(-1) + 29781.555 - 10.816418 * T - 306.098e-25 * T**7
    gibbsTiliq = -19887.066 + 298.7367 * T - 46.29 * T * math.log(T)

    D_gibbs_liq = ((gibbsNbliq + R * T * (math.log(X) + 1.0)) - \
                       gibbsTiliq - R * T * (math.log(1.0 - X) + 1.0) + (1 - 2*X) * 7406.1) / VM
    return D_gibbs_liq

@numba.jit
def DgibbsFCC(X,T,R,VM):
    gibbsNbFCC = -8519.353 + 142.045475 * T - 26.4711 * T * math.log(T) + 0.203475e-3 * T ** 2 - 0.35012e-6 * T ** 3 + \
        93399 * T ** (-1)

    gibbsTiFCC = 26483.26 - 182.426471 * T + 19.0900905 * T * math.log(T) - 22.00832e-3 * T ** 2 + 1.228863e-6 * T ** 3 + \
        1400501 * T ** (-1)

    return ((gibbsNbFCC + R * T * (math.log(X) + 1.0)) - gibbsTiFCC - R *\
                        T * (math.log(1.0 - X) + 1.0) + (1 - 2. * X) * (13045.3)) / VM

@numba.jit
def DDgibbsliq(X,T,R,VM):
    return ((R * T * (1 / X)) + R * T * (1 / (1 - X)) + 2 * 7406.1) / VM

@numba.jit
def eps_theta(tht, k_an, v_an, epsilon,theta_0):
    return epsilon *(1 + (v_an * math.cos(k_an*(tht - theta_0))))

@numba.jit
def eps_theta_prime(tht, k_an, v_an, epsilon):
    return (-k_an * v_an * epsilon * math.sin(k_an * tht))

@cuda.jit
def solvePhi(fields,params,fields_out):
    phi = fields[0]
    c = fields[1]
    c_s = fields[2]
    c_l = fields[3]
    theta = fields[4]
    Tarr = fields[5]
    term1 = fields[6]
    term2 = fields[7]
    term3 = fields[8]
    allterm = fields[9]

    phi_out = fields_out[0]
    c_out = fields_out[1]

    dx = params[0]
    dt = params[1]
    epsilon = params[10]
    w = params[11]
    R = params[12]
    Vm = params[13]
    k_an = params[16]
    v_an = params[17]
    Mphi = params[18]

    theta_0 = 0.0

    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(starty + 1,phi.shape[0] - 1,stridey):
        for j in range(startx + 1,phi.shape[1] -1,stridex):

            T = Tarr[i][j]
            f_S = gibbsFCC(c_s[i][j], T, R, Vm)
            f_L = gibbsliq(c_l[i][j], T, R, Vm)
            mu_S = DgibbsFCC(c_s[i][j], T, R, Vm)
            mu_L = Dgibbsliq(c_l[i][j], T, R, Vm)

            g = phi[i][j] * (1 - phi[i][j])
            h = phi[i][j] ** 2 * (3 - 2 * phi[i][j])
            g_prime = -2 * phi[i][j] + 1
            h_prime = -6 * phi[i][j] ** 2 + 6 * phi[i][j]
            laplacian_phi = (0.5 * (phi[i][j-1] + phi[i][j+1] + phi[i+1][j] + phi[i-1][j] + \
                                    0.5 * (phi[i+1][j+1] + phi[i-1][j+1] + phi[i-1][j-1] + \
                                           phi[i+1][j-1]) - 6 * phi[i][j])) / (dx * dx)

            #
            eps_theta_loc = eps_theta(theta[i][j], k_an, v_an, epsilon,theta_0)
            eps_theta_ip = eps_theta(theta[i+1][j], k_an, v_an, epsilon,theta_0)
            eps_theta_in = eps_theta(theta[i-1][j], k_an, v_an, epsilon,theta_0)
            eps_theta_jp = eps_theta(theta[i][j+1], k_an, v_an, epsilon,theta_0)
            eps_theta_jn = eps_theta(theta[i][j-1], k_an, v_an, epsilon,theta_0)

            eps_theta_prime_ip = eps_theta_prime(theta[i+1][j], k_an, v_an, epsilon)
            eps_theta_prime_in = eps_theta_prime(theta[i-1][j], k_an, v_an, epsilon)
            eps_theta_prime_jp = eps_theta_prime(theta[i][j+1], k_an, v_an, epsilon)
            eps_theta_prime_jn = eps_theta_prime(theta[i][j-1], k_an, v_an, epsilon)


            addterm_x = ((eps_theta_ip * eps_theta_prime_ip * ((phi[i+1][j+1] - phi[i+1][j-1]) / (2*dx))) - \
                (eps_theta_in * eps_theta_prime_in * ((phi[i-1][j+1] - phi[i-1][j-1])/ (2*dx)))) / (2*dx)

            addterm_y = ((eps_theta_jp * eps_theta_prime_jp * ((phi[i+1][j+1] - phi[i-1][j+1]) / (2*dx))) -\
                (eps_theta_jn * eps_theta_prime_jn * ((phi[i+1][j-1] - phi[i-1][j-1])/ (2*dx)))) / (2*dx)
            #
            dphidt = eps_theta_loc ** 2 * laplacian_phi - addterm_x + addterm_y - w * g_prime +\
             (f_L - f_S - (c_l[i][j] * mu_L - c_s[i][j] * mu_S)) * h_prime

            phi_ti_ti = -46640.1607 - 136.5900 * T
            phi_nb_ti = -46640.1607 - 136.62401 * T
            phi0_ti = 0
            phi_nb_nb = -66079.56823 - 138.34327 * T
            phi_ti_nb = -66079.56823 - 138.30922 * T
            phi0_nb = 0

            phi_ti = (1 - c_l[i][j]) * phi_ti_ti + c_l[i][j] * phi_nb_ti + c_l[i][j] * (1 - c_l[i][j]) * (phi0_ti)
            phi_nb = (c_l[i][j]) * phi_nb_nb + (1 - c_l[i][j]) * phi_ti_nb + c_l[i][j] * (1 - c_l[i][j]) * (phi0_nb)
            M_ti = 1 / R / T * math.exp(phi_ti / R / T)
            M_Nb = 1 / R / T * math.exp(phi_nb / R / T)
            Gtiti = R * T / (1 - c_l[i][j])
            Gnbnb = R * T / c_l[i][j]
            Gtinb = 7406.1
            Dl = (1 - c_l[i][j]) * c_l[i][j] * ((1 - c_l[i][j]) * M_Nb + (c_l[i][j]) * (M_ti)) * (Gtiti + Gnbnb - 2 * Gtinb)
            # Dl : the diffusivity matrix in liquid phase
            curr_phi = phi[i][j]
            zeta = (c_l[i][j] - c_s[i][j]) ** 2 * DDgibbsliq(c_l[i][j], T, R, Vm) / Dl #zeta is the missing part of M in(11)

            if zeta < 1e-12:
                phi_out[i][j] = curr_phi
            else:
                phi_out[i][j] = curr_phi + dt * (Mphi / zeta * dphidt) #

            if phi_out[i][j] < 0.0:
                phi_out[i][j] = 0
            elif phi_out[i][j] > 1.0:
                phi_out[i][j] = 1

            term3[i][j] = T
            allterm[i][j] = zeta
            term1[i][j] = c_l[i][j]
            term2[i][j] = c_s[i][j]

#boundary condition##### 注意仅当i+1 等被使用时候才需要进行这一步
    for i in range(starty, phi.shape[0], stridey):
        for j in range(startx, phi.shape[1], stridex):
            if i == 0 or i == phi.shape[0]:
               for j in range(phi.shape[1]):
                  phi_out[0][j] = phi_out[1][j]
                  phi_out[phi.shape[0]][j] = phi_out[phi.shape[0] - 1][j]

            if j == 0 or j == phi.shape[1]:
               for i in range(phi.shape[0]):
                   phi_out[i][0] = phi_out[i][1]
                   phi_out[i][phi.shape[1]] = phi_out[i][phi.shape[1] - 1]



@cuda.jit
def solveC(fields,params,fields_out):
    phi = fields[0]
    c = fields[1]
    c_s = fields[2]
    c_l = fields[3]
    theta = fields[4]
    Tarr = fields[5]
    term1 = fields[6]
    term2 = fields[7]
    term3 = fields[8]
    allterm = fields[9]
    Ds = fields[10]
    Dl = fields[11]

    phi_out = fields_out[0]  #
    c_out = fields_out[1]
    Ds_out = fields_out[10]
    Dl_out = fields_out[11]
    dcdt_out = fields_out[12]

    dx = params[0]
    dt = params[1]
    epsilon = params[10]
    w = params[11]
    R = params[12]

    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    for i in range(starty + 1, phi.shape[0] - 1, stridey):
        for j in range(startx + 1, phi.shape[1] - 1, stridex):

            T = Tarr[i][j]
            #phi next has been calculated in solv_phi
            dphidt_ij = (phi_out[i][j] - phi[i][j]) / dt
            dphidt_irightj = (phi_out[i+1][j] - phi[i+1][j]) / dt
            dphidt_ileftj = (phi_out[i-1][j] - phi[i-1][j]) / dt
            dphidt_ijdown = (phi_out[i][j-1] - phi[i][j-1]) / dt
            dphidt_ijup = (phi_out[i][j+1] - phi[i][j+1]) / dt

            alpha_ij = (epsilon / math.sqrt(2 * w)) * (c_l[i][j] - c_s[i][j]) * \
                       math.sqrt(phi_out[i][j] * (1 - phi_out[i][j]))
            alpha_irightj = (epsilon / math.sqrt(2 * w)) * (c_l[i+1][j] - c_s[i+1][j]) * \
                            math.sqrt(phi_out[i+1][j] * (1 - phi_out[i+1][j]))
            alpha_ileftj = (epsilon / math.sqrt(2 * w)) * (c_l[i-1][j] - c_s[i-1][j]) * \
                           math.sqrt(phi_out[i-1][j] * (1 - phi_out[i-1][j]))
            alpha_ijdown = (epsilon / math.sqrt(2 * w)) * (c_l[i][j-1] - c_s[i][j-1]) * \
                           math.sqrt(phi_out[i][j-1] * (1 - phi_out[i][j-1]))
            alpha_ijup = (epsilon / math.sqrt(2 * w)) * (c_l[i][j+1] - c_s[i][j+1]) * \
                         math.sqrt(phi_out[i][j+1] * (1 - phi_out[i][j+1]))
            #calc D
            phi_ti_ti = -46640.1607 - 136.5900 * T
            phi_nb_ti = -46640.1607 - 136.62401 * T
            phi0_ti = 0
            phi_nb_nb = -66079.56823 - 138.34327 * T
            phi_ti_nb = -66079.56823 - 138.30922 * T
            phi0_nb = 0

            phi_ti = (1 - c_l[i][j]) * phi_ti_ti + c_l[i][j] * phi_nb_ti + c_l[i][j] * (1 - c_l[i][j]) * (phi0_ti)
            phi_nb = (c_l[i][j]) * phi_nb_nb + (1 - c_l[i][j]) * phi_ti_nb + c_l[i][j] * (1 - c_l[i][j]) * (phi0_nb)
            M_ti = 1 / R / T * math.exp(phi_ti / R / T)
            M_Nb = 1 / R / T * math.exp(phi_nb / R / T)
            Gtiti = R * T / (1 - c_l[i][j])
            Gnbnb = R * T / c_l[i][j]
            Gtinb = 7406.1
            Dl_out[i][j] = (1 - c_l[i][j]) * c_l[i][j] * ((1 - c_l[i][j]) * M_Nb + (c_l[i][j]) * (M_ti)) * (
                    Gtiti + Gnbnb - 2 * Gtinb)

            phi_ti_ti = -151989.95 - 127.37 * T
            phi_nb_ti = -369002.77 - 87.15 * T
            phi0_ti = 86711.4 + 2.61 * T
            phi_nb_nb = -395598.95 - 82.03 * T
            phi_ti_nb = -171237.75 - 115.83 * T
            phi0_nb = 107764.17 - 14.52 * T
            phi_ti = (1 - c_s[i][j]) * phi_ti_ti + c_s[i][j] * phi_nb_ti + c_s[i][j] * (
                    1 - c_s[i][j]) * (phi0_ti)
            phi_nb = (c_s[i][j]) * phi_nb_nb + (1 - c_s[i][j]) * phi_ti_nb + c_s[i][j] * (
                    1 - c_s[i][j]) * (phi0_nb)
            M_ti = 1 / R / T * math.exp(phi_ti / R / T)
            M_Nb = 1 / R / T * math.exp(phi_nb / R / T)
            Gtiti = R * T / (1 - c_s[i][j])
            Gnbnb = R * T / c_s[i][j]
            Gtinb = 13045.3
            Ds_out[i][j] = (1 - c_s[i][j]) * c_s[i][j] * ((1 - c_s[i][j]) * M_Nb + (c_s[i][j]) * (M_ti)) * (
                    Gtiti + Gnbnb - 2 * Gtinb)
            #1st term in (7)
            xplus_s = ((phi_out[i][j] * Ds_out[i][j] + phi_out[i+1][j] * Ds_out[i][j]) / 2) * ((c_s[i+1][j] - c_s[i][j]) / dx)  # x05
            xmins_s = ((phi_out[i][j] * Ds_out[i][j] + phi_out[i-1][j] * Ds_out[i][j]) / 2) * ((c_s[i-1][j] - c_s[i][j]) / dx)
            yplus_s = ((phi_out[i][j] * Ds_out[i][j] + phi_out[i][j+1] * Ds_out[i][j]) / 2) * ((c_s[i][j+1] - c_s[i][j]) / dx)
            ymins_s = ((phi_out[i][j] * Ds_out[i][j] + phi_out[i][j-1] * Ds_out[i][j]) / 2) * ((c_s[i][j-1] - c_s[i][j]) / dx)

            xplus_l = ((((1 - phi_out[i][j]) * Dl_out[i][j] + (1 - phi_out[i+1][j]) * Dl_out[i][j]) / 2) * (c_l[i+1][j] - c_l[i][j])) / dx
            xmins_l = ((((1 - phi_out[i][j]) * Dl_out[i][j]+ (1 - phi_out[i-1][j]) * Dl_out[i][j]) / 2) * (c_l[i-1][j] - c_l[i][j])) / dx
            yplus_l = ((((1 - phi_out[i][j]) * Dl_out[i][j] + (1 - phi_out[i][j+1]) * Dl_out[i][j]) / 2) *(c_l[i][j+1] - c_l[i][j])) / dx
            ymins_l = ((((1 - phi_out[i][j]) * Dl_out[i][j]+ (1 - phi_out[i][j-1]) * Dl_out[i][j]) / 2) *(c_l[i][j-1] - c_l[i][j])) / dx

            # antitrapping term
            dxphi_xplus = (phi_out[i+1][j] - phi_out[i][j]) / dx
            dyphi_xplus = (phi_out[i][j+1] - phi_out[i][j-1] + phi_out[i+1][j+1] - phi_out[i+1][j-1]) / (
                        4 * dx)
            # dphi/dj[i05][j] = (phinext[i][j]+ phinext[i+1][j])/2dx
            if math.sqrt(dxphi_xplus**2 + dyphi_xplus**2) < 1e-9:
                xplus_anti = 0.0
            else:
                xplus_anti = (alpha_ij * dphidt_ij + alpha_irightj * dphidt_irightj) * 1 / 2 * \
                             (dxphi_xplus / math.sqrt(dxphi_xplus ** 2 + dyphi_xplus ** 2))

            dxphi_xmins = (phi_out[i][j] - phi_out[i-1][j]) / dx * (-1)
            dyphi_xmins = (phi_out[i-1][j+1] - phi_out[i-1][j-1] + phi_out[i][j+1] - \
                           phi_out[i][j-1]) / (4 * dx) * -1

            if math.sqrt(dxphi_xmins ** 2 + dyphi_xmins ** 2) < 1e-9:
                xmins_anti = 0.0
            else:
                xmins_anti = (alpha_ileftj * dphidt_ileftj + alpha_ij * dphidt_ij) * 1 / 2 * \
                            (dxphi_xmins / math.sqrt(dxphi_xmins ** 2 + dyphi_xmins ** 2))


            dxphi_yplus = (phi_out[i][j+1] - phi_out[i][j]) / dx
            dyphi_yplus = (phi_out[i+1][j] - phi_out[i-1][j] + phi_out[i+1][j+1] - \
                           phi_out[i-1][j+1]) / (4 * dx)

            if math.sqrt(dxphi_yplus ** 2 + dyphi_yplus ** 2) < 1e-9:
                yplus_anti = 0.0
            else:
                yplus_anti = (alpha_ij * dphidt_ij + alpha_ijup * dphidt_ijup) * 1 / 2 * \
                             (dxphi_yplus / math.sqrt(dxphi_yplus ** 2 + dyphi_yplus ** 2))


            dxphi_ymins = (phi_out[i][j] - phi_out[i][j-1]) / dx * (-1)
            dyphi_ymins = (phi_out[i+1][j-1] - phi_out[i-1][j-1] + phi_out[i+1][j] - \
                           phi_out[i-1][j]) / (4 * dx) * (-1)

            if math.sqrt(dxphi_ymins ** 2 + dyphi_ymins ** 2) < 1e-9:
                ymins_anti = 0.0
            else:
                ymins_anti = (alpha_ijdown * dphidt_ijdown + alpha_ij * dphidt_ij) * 1 / 2 *\
                             (dxphi_ymins / math.sqrt(dxphi_ymins ** 2 + dyphi_ymins ** 2))


            # dc/dt completed term
            dcdt_out[i][j] = (xplus_s + xmins_s + yplus_s + ymins_s) / dx + (xplus_l + xmins_l + yplus_l + ymins_l) / dx + \
                   (xplus_anti + xmins_anti + yplus_anti + ymins_anti) / dx


            c_out[i][j] = c[i][j] + dt * dcdt_out[i][j] #


            if c_out[i][j] < 0.0:
                c_out[i][j] = 0.0000001
            elif c_out[i][j] > 1.0:
                c_out[i][j] = 0.9999999

@cuda.jit
def updateAll(fields,params,fields_out):#update phi,c,c_l,c_s,k,theta
    phi = fields[0]
    c = fields[1]
    c_s = fields[2]
    c_l = fields[3]
    theta = fields[4]
    Tarr = fields[5]

    phi_out = fields_out[0]  #
    c_out = fields_out[1]
    c_s_out = fields_out[2]
    c_l_out = fields_out[3]
    theta_out = fields_out[4]
    Tarr_out = fields_out[5]

    dx = params[0]
    dt = params[1]
    r = params[2]
    R = params[12]
    VM = params[13]
    kini = params[14]

    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    for i in range(starty+ 1, phi.shape[0]- 1, stridey):
        for j in range(startx + 1, phi.shape[1]- 1, stridex):

            T = Tarr[i][j]
            #注意仅针对在边界上，k值会发生变化
            if phi_out[i][j] > 0.0 and phi_out[i][j] < 1.0:
                curr_k = c_s[i][j] / c_l[i][j]
                mu_L = Dgibbsliq(c_l[i][j], T, R, VM)
                mu_S = DgibbsFCC(c_s[i][j], T, R, VM)

                kdot = r * (mu_L - mu_S)
                k = curr_k + kdot * dt
            else:
                k = kini

            c_s_out[i][j] = c_out[i][j] * k / (k * phi_out[i][j] - phi_out[i][j] + 1) #
            c_l_out[i][j] = c_out[i][j] / (k * phi_out[i][j] - phi_out[i][j] + 1) #
            
        #what is double obstacle?
            if c_s_out[i][j] < 0.0:
                c_s_out[i][j] = 0.0000001
            elif c_s_out[i][j] > 1.0:
                c_s_out[i][j] = 0.9999999

            if c_l_out[i][j] < 0.0:
                c_l_out[i][j] = 0.0000001
            elif c_l_out[i][j] > 1.0:
                c_l_out[i][j] = 0.9999999

    for i in range(starty + 1, phi.shape[0] - 1, stridey):
        for j in range(startx + 1, phi.shape[1] - 1, stridex):

            dphidy = (phi_out[i][j+1] - phi_out[i][j-1]) / (2 * dx)
            dphidx = (phi_out[i+1][j] - phi_out[i-1][j]) / (2 * dx)
          
            if (dphidy == 0.0) and (dphidx == 0.0):
                theta_out[i][j] = 0.0
            else:
                theta_out[i][j] = np.arctan2(dphidy,dphidx)

#BC
@cuda.jit
def TarrUpdate(fields,params,fields_out,timestep):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    Tarr = fields[5]
    Tarr_out = fields_out[5]

    dx = params[0]
    dt = params[1]
    Vs = params[3]
    G = params[4]
    T0 = params[7]
    xoffs = params[20]
    t = timestep * dt
    for i in range(starty, Tarr.shape[0], stridey):
        for j in range(startx, Tarr.shape[1], stridex):

            #Tarr_out[i][j] = (T0 + G * ((j-1) * dx + xoffs - Vs * t))
            Tarr_out[i][j] = round((T0 + G * ((j - 1) * dx + xoffs - Vs * t)),4)

@cuda.jit
def initBC(fields):
    phi = fields[0]
    c = fields[1]
    c_s = fields[2]
    c_l = fields[3]
    theta = fields[4]

    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(starty,phi.shape[0],stridey):
        for j in range(startx,phi.shape[1],stridex):
            ##### for ic initiate
            if i == 0 or i == phi.shape[0]:
                for j in range(phi.shape[1]):
                    phi[0][j] = phi[1][j]
                    phi[phi.shape[0]][j] = phi[phi.shape[0] - 1][j]
                    c[0][j] = c[1][j]
                    c[phi.shape[0]][j] = c[phi.shape[0] - 1][j]
                    c_l[0][j] = c_l[1][j]
                    c_l[phi.shape[0]][j] = c_l[phi.shape[0] - 1][j]
                    c_s[0][j] = c_s[1][j]
                    c_s[phi.shape[0]][j] = c_s[phi.shape[0] - 1][j]
                    theta[0][j] = theta[1][j]
                    theta[phi.shape[0]][j] = theta[phi.shape[0] - 1][j]

            if j == 0 or j == phi.shape[1]:
                for i in range(phi.shape[0]):
                    phi[i][0] = phi[i][1]
                    phi[i][phi.shape[1]] = phi[i][phi.shape[1] - 1]
                    c[i][0] = c[i][1]
                    c[i][phi.shape[1]] = c[i][phi.shape[1] - 1]
                    c_s[i][0] = c_s[i][1]
                    c_s[i][phi.shape[1]] = c_s[i][phi.shape[1] - 1]
                    c_l[i][0] = c_l[i][1]
                    c_l[i][phi.shape[1]] = c_l[i][phi.shape[1] - 1]
                    theta[i][0] = theta[i][1]
                    theta[i][phi.shape[1]] = theta[i][phi.shape[1] - 1]

@cuda.jit
def PullBack(fields,params,fields_out):

    phi = fields[0]
    c = fields[1]
    c_s = fields[2]
    c_l = fields[3]
    theta = fields[4]
    Tarr = fields[5]

    phi_out = fields_out[0]
    c_out = fields_out[1]
    c_s_out = fields_out[2]
    c_l_out = fields_out[3]
    theta_out = fields_out[4]
    Tarr_out = fields_out[5]

    M = Tarr.shape[1]
    c_0 = params[15]
    kini = params[14]
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(starty + 1, phi.shape[0] - 1, stridey):
        for j in range(startx + 1, phi.shape[1]- 1, stridex):

            if j < 2000 :
                phi_out[i][j] = phi_out[i][j + 1]
                c_out[i][j] = c_out[i][j + 1]
                c_l_out[i][j] = c_l_out[i][j + 1]
                c_s_out[i][j] = c_s_out[i][j + 1]
                theta_out[i][j] = theta_out[i][j + 1]

            else:
                phi_out[i][j] = 0
                c_out[i][j] = c_0
                c_s_out[i][j] = c[i,j] * kini / (kini * phi[i,j] - phi[i,j] + 1) #c_s_out[i][M-2]  # c_s[i, M-1]
                c_l_out[i][j] = c[i,j] / (kini * phi[i,j] - phi[i,j] + 1) #c_l_out[i][M-2]  # c_l[i, M-1]
                theta_out[i][j] = 0# theta_out[i][j-1]

            cuda.syncthreads()

class pcr_with_mf(Simulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses_gpu = True
        self._framework = "GPU_SERIAL"

    def init_tdb_params(self):
        super().init_tdb_params()

    def init_fields(self):
        dim = self.dimensions
        N = dim[0]
        M = dim[1]
        phi = np.zeros(dim)
        c = np.zeros(dim)
        c_s = np.ones(dim)
        c_l = np.ones(dim)
        theta = np.zeros(dim)
        Tarr = np.zeros(dim)
        term1 = np.zeros(dim)
        term2 = np.zeros(dim)
        term3 = np.zeros(dim)
        allterm = np.zeros(dim)
        Ds = np.zeros(dim)
        Dl = np.zeros(dim)
        dcdt = np.zeros(dim)

        Tl= self.user_data["Tl"]
        xi = self.user_data["xi"]
        dx = self.dx
        initXpos = self.user_data["initXpos"]
        T0 = self.user_data["T0"]
        G = self.user_data["G"]
        kini = self.user_data["kini"]
        c_0 = self.user_data["c_0"]
        # init Tarr
        for j in range(M):
            value = T0 + G * (j * dx)
            for i in range(N):
                Tarr[i][j] = value

        #init phi
        posArr_h = initXpos * dx + (np.random.rand(N) - 0.5) * dx # 1D array
        posArr = np.zeros(N)
        posArr[:] = posArr_h

        for i in range(N):
            for j in range(M):
                x_p = j * dx
                xpos = posArr[i]

                phi[i][j] = ((1.0 - np.tanh((x_p - xpos) / xi)) / 2)
                if phi[i][j] > 1.0:
                    phi[i][j] = 1.0

        #init c
        for i in range(N):
            for j in range(M):
                h = phi[i][j] ** 2 * (3 - 2 * phi[i][j])
                c[i][j] = round((h * kini * c_0 + (1 - h) * c_0),3)

        #init c_s and c_l
        c_s = c_s * c_0 * kini
        c_l = c_l * c_0

        #init theta
        for i in range(0,N-1):
            for j in range(1,M-1):

                dphidy = (phi[i][j+1] - phi[i][j-1]) / (2 * dx)
                dphidx = (phi[i+1][j] - phi[i-1][j]) / (2 * dx)
                if (dphidy == 0.0) and (dphidx == 0.0):
                    theta[i][j] = 0.0
                else:
                    theta[i][j] = np.arctan2(dphidy,dphidx)

        #注意这里的bc 似乎不需要
        for j in range(M):
            theta[0][j] = theta[1][j]
            theta[M-1][j] = theta[M-2][j]
        for i in range(N):
            theta[i][0] = theta[i][1]
            theta[i][N-1] = theta[i][N-2]

        self.add_field(phi, "phi",colormap=COLORMAP_PHASE)
        self.add_field(c, "c")
        self.add_field(c_s,"c_s")
        self.add_field(c_l, "c_l")
        self.add_field(theta, "theta")
        self.add_field(Tarr,"Tarr")
        self.add_field(term1, "term1")
        self.add_field(term2, "term2")
        self.add_field(term3, "term3")
        self.add_field(allterm, "allterm")
        self.add_field(Ds, "Ds")
        self.add_field(Dl, "Dl")
        self.add_field(dcdt, "dcdt")


    def just_before_simulating(self):  #in simulation file
        super().just_before_simulating()

        params = []
        params.append(self.dx)
        params.append(self.dt)
        params.append(self.user_data["r"])
        params.append(self.user_data["Vs"])
        params.append(self.user_data["G"])
        params.append(self.user_data["dT"])
        params.append(self.user_data["initXpos"])
        params.append(self.user_data["T0"])
        params.append(self.user_data["xi"])
        params.append(self.user_data["sigma"])
        params.append(self.user_data["epsilon"])
        params.append(self.user_data["w"])
        params.append(self.user_data["R"])
        params.append(self.user_data["VM"])
        params.append(self.user_data["kini"])
        params.append(self.user_data["c_0"])
        params.append(self.user_data["k_an"])
        params.append(self.user_data["v_an"])
        params.append(self.user_data["Mphi"])
        params.append(self.user_data["Tl"])
        params.append(self.user_data["xoffs"])

        self.user_data["params"] = np.array(params)
        self.user_data["params_GPU"] = cuda.to_device(self.user_data["params"])


    def simulation_loop(self):

        cuda.synchronize()

        solvePhi[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)

        cuda.synchronize()
        solveC[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)

        cuda.synchronize()
        updateAll[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)

        cuda.synchronize()
        TarrUpdate[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device,self.time_step_counter -1)
        cuda.synchronize()

        thresh = int(0.5*2000)
        phi = self._fields_out_gpu_device[0]
        phi_column_j1000 = [phi[i][thresh] for i in range(len(phi))]
        maxphi_thresh = max(phi_column_j1000)
        if maxphi_thresh > 0.1:
            PullBack[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)
            self.user_data["params_GPU"][20] += self.dx
            cuda.synchronize()


        if (self.time_step_counter -1) % 500000                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        == 0:
            self.retrieve_fields_from_GPU()
            output_folder = 'mf'  ##
            os.makedirs(output_folder, exist_ok=True)
            phi = self.fields[0].data
            df = pd.DataFrame(phi)
            filename = os.path.join(output_folder, f'phi_step_{self.time_step_counter -1}.csv')
            df.to_csv(filename, index=False, header=False)

            c = self.fields[1].data
            df = pd.DataFrame(c)
            filename = os.path.join(output_folder, f'c_step{self.time_step_counter -1 }.csv')
            df.to_csv(filename, index=False, header=False)

            tarr = self.fields[5].data
            df = pd.DataFrame(tarr)
            filename = os.path.join(output_folder, f'tarr_step{self.time_step_counter -1 }.csv')
            df.to_csv(filename, index=False, header=False)

            #theta = self.fields[4].data
            #df = pd.DataFrame(theta)
            #filename = os.path.join(output_folder, f'theta_step{self.time_step_counter -1 }.csv')
            #df.to_csv(filename, index=False, header=False)








