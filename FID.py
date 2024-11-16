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
   
    gibbs_liq = (R * T * (X * math.log(X) + (1.0 - X) * math.log(1.0 - X))) / VM
    return gibbs_liq

@numba.jit
def gibbsFCC(X,T,R,VM,ke,me,Tm):
   
    gibbs_FCC = (R * T / VM) * (X * math.log(X) + (1.0 - X) * math.log(1.0 - X) - X * math.log(ke) + 
    (1.0 - X) * math.log((1.0 + ((Tm - T) / me)) / (1.0 + ((ke * (Tm - T)) / me)))
)
    return gibbs_FCC

@numba.jit
def Dgibbsliq(X,T,R,VM):
    
    D_gibbs_liq = (R * T * ( math.log(X) - math.log(1.0 - X))) / VM
    return D_gibbs_liq

@numba.jit
def DgibbsFCC(X,T,R,VM,ke,me,Tm):
    result = -(R * T * (math.log((me + Tm - T) / (ke * Tm - ke * T + me)) + math.log(ke) + math.log(1-X) -math.log(X))) / VM

    return result


@cuda.jit
def solvePhi(fields,params,fields_out):
    phi = fields[0]
    c = fields[1]
    c_s = fields[2]
    c_l = fields[3]
    Tarr = fields[4]
    beta = fields[5]

    phi_out = fields_out[0]
    c_out = fields_out[1]

    dx = params[0]
    dt = params[1]
    mu = params[3]
    P = params[2]
    sigma = params[8]
    w = params[11]
    R = params[12]
    VM = params[10]
    ke = params[12]
    c_0 = params[13]
    me = params[14]
    k_an = params[15]
    v_an = params[16]
    Tm = params[17]
   
    xi = params[22] 
 

    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(starty + 1,phi.shape[0] - 1,stridey):
        for j in range(startx + 1,phi.shape[1] -1,stridex):

            T = Tarr[i][j]
            f_S = gibbsFCC(c_s[i][j], T, R, VM,ke, me, Tm)
            f_L = gibbsliq(c_l[i][j], T, R, VM)
            mu_S = DgibbsFCC(c_s[i][j], T, R, VM,ke, me, Tm)
            mu_L = Dgibbsliq(c_l[i][j], T, R, VM)

            dphidy = (phi[i][j+1]-phi[i][j-1])/(2*dx)
            dphidx = (phi[i+1][j]-phi[i-1][j])/(2*dx)
            normdenom = math.sqrt(dphidx**2 + dphidy**2)

            if normdenom > 1e-9:
                normx = dphidx/normdenom
                normy = dphidy/normdenom
            else:
                normx = 0.0
                normy = 0.0

            sigma_n = sigma * (1 - v_an * (3 - 4 * (normx**4 + normy**4)))
            mu_n = mu * (1 - v_an * (3 - 4 * (normx**4 + normy**4)))
           
            #Calculate K (kinetic coefficient) and dG (driving force)
            K = 8 * P * xi * mu_n / (8 * P * xi + mu_n * math.pi**2 * (c_s[i][j] - c_l[i][j])**2)

            dG = f_L - f_S - (phi[i][j]*mu_S + (1-phi[i][j])*mu_L)*(c_l[i][j]-c_s[i][j])
            laplacian_phi = (0.5 * (phi[i][j-1] + phi[i][j+1] + phi[i+1][j] + phi[i-1][j] + \
                                                0.5 * (phi[i+1][j+1] + phi[i-1][j+1] + phi[i-1][j-1] + \
                                                    phi[i+1][j-1]) - 6 * phi[i][j])) / (dx * dx)
            

            dphidt = K * (sigma_n * (laplacian_phi + math.pi**2 / xi**2 * (phi[i][j] - 0.5)) + math.pi/xi * math.sqrt(phi[i][j] * (1-phi[i][j])) * dG)
            curr_phi = phi[i][j]

            eta = 0.01
            betaval = beta[i][j] - 0.5
          
            phi_out[i,j] = curr_phi + dt*dphidt + eta*betaval*math.sqrt(dt)

            if phi_out[i][j] < 0.0:
                phi_out[i][j] = 0
            elif phi_out[i][j] > 1.0:
                phi_out[i][j] = 1



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
    Tarr = fields[4]
 
    phi_out = fields_out[0]  #
    c_out = fields_out[1]
    c_s_out = fields[2]
    c_l_out = fields[3]
   
    dx = params[0]
    dt = params[1]
    mu = params[3]
    P = params[2]
    sigma = params[8]
    w = params[11]
    R = params[12]
    VM = params[10]
    ke = params[12]
    c_0 = params[13]
    me = params[14]
    k_an = params[15]
    v_an = params[16]
    Tm = params[17]
   
    xi = params[22] 
  
    Ds = params[21]
    Dl = params[20]
    

    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    for i in range(starty + 1, phi.shape[0] - 1, stridey):
        for j in range(startx + 1, phi.shape[1] - 1, stridex):

            T = Tarr[i][j]
            #phi next has been calculated in solv_phi
            dphidt_ij = (phi_out[i][j] - phi[i][j]) / dt
            dphidt_opp = ((1-phi_out[i,j]) - (1-phi[i,j])) / dt

            mu_S = DgibbsFCC(c_s[i][j], T,R,VM,ke,me,Tm)
            mu_L = Dgibbsliq(c_l[i][j], T, R, VM)

            xplus_s = ((phi_out[i][j] * Ds + phi_out[i+1][j] * Ds) / 2) * ((c_s[i+1][j] - c_s[i][j]) / dx)
            xmins_s = ((phi_out[i][j] * Ds + phi_out[i-1][j] * Ds) / 2) * ((c_s[i-1][j] - c_s[i][j]) / dx)
            yplus_s = ((phi_out[i][j] * Ds + phi_out[i][j+1] * Ds) / 2) * ((c_s[i][j+1] - c_s[i][j]) / dx)
            ymins_s = ((phi_out[i][j] * Ds + phi_out[i][j-1] * Ds) / 2) * ((c_s[i][j-1] - c_s[i][j]) / dx)

            #liquid component of diffusion
            xplus_l = ((((1-phi_out[i][j]) * Dl + (1-phi_out[i+1][j]) * Dl) / 2) * (c_l[i+1][j] - c_l[i][j]) / dx)
            xmins_l = ((((1-phi_out[i][j]) * Dl + (1-phi_out[i-1][j]) * Dl) / 2) * (c_l[i-1][j] - c_l[i][j]) / dx)
            yplus_l = ((((1-phi_out[i][j]) * Dl + (1-phi_out[i][j+1]) * Dl) / 2) * (c_l[i][j+1] - c_l[i][j]) / dx)
            ymins_l = ((((1-phi_out[i][j]) * Dl + (1-phi_out[i][j-1]) * Dl) / 2) * (c_l[i][j-1] - c_l[i][j]) / dx)
            
            #dc/dt completed term
            if phi_out[i][j] < 1e-3: #if we are in liquid portion
                dcsdt = 0.0
                dcldt = (xplus_l + xmins_l + yplus_l + ymins_l) / dx
            elif phi_out[i][j] > 1-1e-3: #if we are in solid portion
                dcsdt = (xplus_s + xmins_s + yplus_s + ymins_s) / dx
                dcldt = 0.0
            else: #if(phi_out[i,j] >= 1e-9 && phi_out[i,j] <= 1-1e-9) #if we are at the interface solve both equations
                dcsdt = ((xplus_s + xmins_s + yplus_s + ymins_s) / dx + P*phi_out[i][j]*(1-phi_out[i][j])*(mu_L - mu_S) + phi_out[i][j] * dphidt_ij * (c_l[i][j] - c_s[i][j])) / phi_out[i][j]
                dcldt = ((xplus_l + xmins_l + yplus_l + ymins_l) / dx + P*phi_out[i][j]*(1-phi_out[i][j])*(mu_S - mu_L) + (1-phi_out[i][j]) * dphidt_opp * (c_s[i][j] - c_l[i,j])) / (1-phi_out[i][j])
        

            c_l_out[i][j] = c_l[i][j] + dt * dcldt
            c_s_out[i][j] = c_s[i][j] + dt * dcsdt
    
            if c_s_out[i][j] < 0.0:
                c_s_out[i][j] = 0.0000001
            elif c_s_out[i][j] > 1.0:
                c_s_out[i][j] = 0.9999999
            
            if c_l_out[i][j] < 0.0:
                c_l_out[i][j] = 0.0000001
            elif c_l_out[i][j] > 1.0:
                c_l_out[i][j] = 0.9999999
          
#BC
@cuda.jit
def TarrUpdate(fields,params,fields_out,timestep):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    Tarr = fields[5]
    Tarr_out = fields_out[5]

    dx = params[0]
    dt = params[1]
    Vs = params[4]
    G = params[5]
    Ts = params[18]
    xoffs = params[19]
    t = timestep * dt
    for i in range(starty, Tarr.shape[0], stridey):
        for j in range(startx, Tarr.shape[1], stridex):

            #Tarr_out[i][j] = (T0 + G * ((j-1) * dx + xoffs - Vs * t))
            Tarr_out[i][j] = round((Ts + G * ((j - 1) * dx + xoffs - Vs * t)),4)

@cuda.jit
def initBC(fields):
    phi = fields[0]
    c = fields[1]
    c_s = fields[2]
    c_l = fields[3]

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
                   

@cuda.jit
def PullBack(fields,params,fields_out):

    phi = fields[0]
    c = fields[1]
    c_s = fields[2]
    c_l = fields[3]
  
    Tarr = fields[4]

    phi_out = fields_out[0]
    c_out = fields_out[1]
    c_s_out = fields_out[2]
    c_l_out = fields_out[3]
 
    Tarr_out = fields_out[4]

    M = Tarr.shape[1]
    c_0 = params[15]
    kini = params[14]
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)

    for i in range(starty + 1, phi.shape[0] - 1, stridey):
        for j in range(startx + 1, phi.shape[1]- 1, stridex):

            if j < 3000 :
                phi_out[i][j] = phi_out[i][j + 1]
                #c_out[i][j] = c_out[i][j + 1]
                c_l_out[i][j] = c_l_out[i][j + 1]
                c_s_out[i][j] = c_s_out[i][j + 1]
             

            else:
                phi_out[i][j] = 0
               
                c_s_out[i][j] = c_0 * kini #c_s_out[i][M-2]  # c_s[i, M-1]
                c_l_out[i][j] = c_0 #c_l_out[i][M-2]  # c_l[i, M-1]
                

            cuda.syncthreads()

class FID(Simulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses_gpu = True
        self._framework = "GPU_SERIAL"

    def init_tdb_params(self):
        super().init_tdb_params()

    def init_fields(self):
        dim = self.dimensions
        N = dim[0] #2000
        M = dim[1] #3000
        phi = np.zeros(dim)
        c = np.zeros(dim)
        c_s = np.ones(dim)
        c_l = np.ones(dim)
        Tarr = np.zeros(dim)

        beta = np.random.rand(N,M).astype(np.float32) #
        Ds = np.zeros(dim)
        Dl = np.zeros(dim)
        dcdt = np.zeros(dim)

        xi = self.user_data["xi"]
        dx = self.dx
        initXpos = self.user_data["initXpos"]
        Ts = self.user_data["Ts"]
        G = self.user_data["G"]
        kini = self.user_data["kini"]
        c_0 = self.user_data["c_0"]
        # init Tarr
        # 创建 Tarr 的二维数组
        Tarr = Ts + G * (np.arange(M) * dx - initXpos * dx)  # 生成一维数组
        Tarr = np.tile(Tarr, (N, 1))  # 将 Tarr 一维数组复制成 N 行的二维数组

        #init phi
        posArr_h = initXpos * dx + (np.random.rand(N) - 0.5) * dx # 1D array
        posArr = np.zeros(N)
        posArr[:] = posArr_h

        x_p = np.arange(M) * dx  # 生成所有的 x_p
        phi = (1.0 - np.tanh((x_p - posArr[:, None]) / xi)) / 2  # 矢量化计算 phi

        # 确保 phi 中的值不超过 1.0
        phi = np.minimum(phi, 1.0)


        #init c_s and c_l
        c_s = c_s * c_0 * kini
        c_l = c_l * c_0

      
        self.add_field(phi, "phi",colormap=COLORMAP_PHASE)
        self.add_field(c, "c")
        self.add_field(c_s,"c_s")
        self.add_field(c_l, "c_l")
        self.add_field(Tarr,"Tarr")

        self.add_field(beta,"beta")
  
        

    def just_before_simulating(self):  #in simulation file
        super().just_before_simulating()

        params = []
        params.append(self.dx)
        params.append(self.dt)
        params.append(self.user_data["P"]) #2
        params.append(self.user_data["mu"]) #3
        params.append(self.user_data["Vs"]) #4
        params.append(self.user_data["G"]) #5
        params.append(self.user_data["dT"]) #6
        params.append(self.user_data["initXpos"]) #7
        params.append(self.user_data["sigma"]) #8
        
        params.append(self.user_data["R"]) #9
        params.append(self.user_data["VM"]) #10
        params.append(self.user_data["kini"])#12
        params.append(self.user_data["c_0"])#13
        params.append(self.user_data["me"])#14
        params.append(self.user_data["k_an"])#15
        params.append(self.user_data["v_an"])#16
        params.append(self.user_data["Tm"])#17
        params.append(self.user_data["Ts"])#18
        
        params.append(self.user_data["xoffs"])#19
        
        params.append(self.user_data["Dl"])#20
        params.append(self.user_data["Ds"])#21

        params.append(self.user_data["xi"])#22

        self.user_data["params"] = np.array(params)
        self.user_data["params_GPU"] = cuda.to_device(self.user_data["params"])


    def simulation_loop(self):

        cuda.synchronize()

        solvePhi[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)

        cuda.synchronize()
        solveC[self._gpu_blocks_per_grid_2D, self._gpu_threads_per_block_2D](self._fields_gpu_device,self.user_data["params_GPU"],self._fields_out_gpu_device)

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

            c_s = self.fields[2].data
            df = pd.DataFrame(c_s)
            filename = os.path.join(output_folder, f'c_s_step{self.time_step_counter -1 }.csv')
            df.to_csv(filename, index=False, header=False)

            tarr = self.fields[4].data
            df = pd.DataFrame(tarr)
            filename = os.path.join(output_folder, f'tarr_step{self.time_step_counter -1 }.csv')
            df.to_csv(filename, index=False, header=False)

        








