#FID implimentation 
#Author: Jose Mancias - December 2023

using DelimitedFiles, CUDA, LinearAlgebra, Random, CUDA.CURAND

# The primary kernal for calculating an iteration of a phase field calculation
function solvePhi(phi::CuDeviceMatrix{Float64,1}, phinext::CuDeviceMatrix{Float64,1}, c_s::CuDeviceMatrix{Float64,1}, 
        c_l::CuDeviceMatrix{Float64,1}, N::Int64, M::Int64, Tarr::CuDeviceVector{Float64, 1}, 
        dx::Float64, R::Float64, Vm::Float64, dt::Float64, P::Float64, mu::Float64, xi::Float64, 
        sigma::Float64, v_an::Float64, ke::Float64, me::Int64, Tm::Float64, beta::CuDeviceMatrix{Float32,1})

  #free energy per volume for two phases
function gibbsliq(x, T, R, Vm)
  return (R * T/Vm) * (x*log(x) + (1.0-x)*log(1.0-x))
end

function Dgibbsliq(x, T, R, Vm)
  return (R * T / Vm) * (log(x) - log(1.0 - x))
end

function gibbsFCC(x, T, R, Vm)
  return (R*T/Vm) *(x*log(x) + (1.0-x)*log(1.0-x) - x*log(ke) + 
            (1.0-x)*log((1.0 + ((Tm-T)/(me)))/ (1.0 + ((ke*(Tm-T))/(me)))))
end

function DgibbsFCC(x, T, R, Vm)
  return -(R *T *(log((me + Tm - T)/(ke* Tm - ke* T + me)) + log(ke) + log(1 - x) - log(x)))/Vm
end


  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

  if i < N && j < M && i > 1 && j > 1

    T = Tarr[j] #Temperature is determined by y position

    # # Thermodynamics
    f_S = gibbsFCC(c_s[i,j], T, R, Vm)
    f_L = gibbsliq(c_l[i,j], T, R, Vm)

    mu_S = DgibbsFCC(c_s[i,j], T, R, Vm)
    mu_L = Dgibbsliq(c_l[i,j], T, R, Vm)

    # # Anisotopy effects -------------------------------------------------------------------
    dphidy = (phi[i, j+1]-phi[i, j-1])/(2*dx)
    dphidx = (phi[i+1, j]-phi[i-1, j])/(2*dx)
    normdenom = sqrt(dphidx^2 + dphidy^2)

    if normdenom > 1e-9
      normx = dphidx / normdenom
      normy = dphidy / normdenom
    else
      normx = 0.0
      normy = 0.0
    end

    sigma_n = sigma * (1 - v_an * (3 - 4 * (normx^4 + normy^4)))
    mu_n = mu * (1 - v_an * (3 - 4 * (normx^4 + normy^4)))

    #c_s = c_alpha,,, c_l = c_beta
    #Calculate K (kinetic coefficient) and dG (driving force)
    K = 8 * P * xi * mu_n / (8 * P * xi + mu_n * pi^2 * (c_s[i,j] - c_l[i,j])^2)

    dG = f_L - f_S - (phi[i,j]*mu_S + (1-phi[i,j])*mu_L)*(c_l[i,j]-c_s[i,j])

    #calculate laplacian
    # laplacian_phi = (phi[i+1, j]+phi[i-1, j]+phi[i, j-1]+phi[i, j+1]-4.0*phi[i,j])/(dx*dx)
    laplacian_phi = (0.5*(phi[i, j-1]+phi[i, j+1]+phi[i+1, j]+phi[i-1, j]+
          0.5*(phi[i+1,j+1] + phi[i-1,j+1] + phi[i-1,j-1] + phi[i+1,j-1]) - 6*phi[i,j])) / (dx*dx)

    #laplacian_phi = (0.5*(phi[i, j-1]+phi[i, j+1]+phi[i+1, j]+phi[i-1, j]+
    #      0.25*(phi[i+1,j+1] + phi[i-1,j+1] + phi[i-1,j-1] + phi[i+1,j-1]) - 3*phi[i,j])) / (dx*dx)

    # Calculate the next phi value according to eq 6 in relaxation paper

    #dphidt = K * (sigma_n * (laplacian_phi + pi^2 / xi^2 * (phi[i,j] - 0.5)) - pi^2/(8*xi) * dG)
    dphidt = K * (sigma_n * (laplacian_phi + pi^2 / xi^2 * (phi[i,j] - 0.5)) + pi/xi * sqrt(phi[i,j] * (1-phi[i,j])) * dG) #more computationally stable version of equation

    #update Phi
    curr_phi = phi[i,j]

    #Noise generation
    eta = 0.01 #noise amplitude

    #Random noise
    # state = CUDA.CURAND.curandState_t() #Declare curand state
    # CUDA.CURAND.curand_init(0, i+ (j-1) * N, 0, state) #seed, squence, offset, state
    # beta = CUDA.CURAND.curand_uniform(state) #uniform random number between 0 and 1

    betaval = beta[i,j] - 0.5 #shift to -0.5, 0.5

    @inbounds phinext[i,j] = curr_phi + dt*dphidt + eta*betaval*sqrt(dt)

    #double obstacle bounds
    if phinext[i,j] < 0.0
      phinext[i,j] = 0.0
    elseif phinext[i,j] > 1.0
      phinext[i,j] = 1.0
    end

  end


  #---
  #---boundary conditions---
  #---

  # periodic conditions on the sides
  if i == 1 && j <= M
    phinext[1, j] = phinext[N-1, j]
  end
  if i == N && j <= M
    phinext[N, j] = phinext[2, j]
  end

  # no-flux conditions on the top and bottom
  if j == 1 && i <= N
    phinext[i, 1] = phinext[i, 2]
  end

  if j == M && i <= N
    phinext[i, M] = phinext[i, M-1]
  end

 
  return 
end

#Solve the diffusion equation in one kernel
function SolveC(phinext::CuDeviceMatrix{Float64,1}, phi::CuDeviceMatrix{Float64,1}, c_s::CuDeviceMatrix{Float64,1}, c_l::CuDeviceMatrix{Float64,1}, 
    csnext::CuDeviceMatrix{Float64,1}, clnext::CuDeviceMatrix{Float64,1}, Vm::Float64, Tarr::CuDeviceVector{Float64,1}, R::Float64, 
    N::Int64, M::Int64, dx::Float64, dt::Float64, P::Float64, Ds::Float64, Dl::Float64, me::Int64, Tm::Float64, ke::Float64)

  
  function Dgibbsliq(x, T, R, Vm)
    return (R * T / Vm) * (log(x) - log(1.0 - x))
  end
  
  
  function DgibbsFCC(x, T, R, Vm)
    return -(R *T *(log((me + Tm - T)/(ke* Tm - ke* T + me)) + log(ke) + log(1 - x) - log(x)))/Vm
  end

  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

  if i < N && j < M && i > 1 && j > 1

    T = Tarr[j]

    #calculate dphidt terms
    dphidt_ij = (phinext[i,j] - phi[i,j]) / dt
    dphidt_opp = ((1-phinext[i,j]) - (1-phi[i,j])) / dt

    #Thermodynamics calculations
    mu_S = DgibbsFCC(c_s[i,j], T, R, Vm)
    mu_L = Dgibbsliq(c_l[i,j], T, R, Vm)

    #calculate dc / dt -------
    #solid component of diffusion
    xplus_s = ((phinext[i,j] * Ds + phinext[i+1,j] * Ds) / 2) * ((c_s[i+1,j] - c_s[i,j]) / dx)
    xmins_s = ((phinext[i,j] * Ds + phinext[i-1,j] * Ds) / 2) * ((c_s[i-1,j] - c_s[i,j]) / dx)
    yplus_s = ((phinext[i,j] * Ds + phinext[i,j+1] * Ds) / 2) * ((c_s[i,j+1] - c_s[i,j]) / dx)
    ymins_s = ((phinext[i,j] * Ds + phinext[i,j-1] * Ds) / 2) * ((c_s[i,j-1] - c_s[i,j]) / dx)

    #liquid component of diffusion
    xplus_l = ((((1-phinext[i,j]) * Dl + (1-phinext[i+1,j]) * Dl) / 2) * (c_l[i+1,j] - c_l[i,j]) / dx)
    xmins_l = ((((1-phinext[i,j]) * Dl + (1-phinext[i-1,j]) * Dl) / 2) * (c_l[i-1,j] - c_l[i,j]) / dx)
    yplus_l = ((((1-phinext[i,j]) * Dl + (1-phinext[i,j+1]) * Dl) / 2) * (c_l[i,j+1] - c_l[i,j]) / dx)
    ymins_l = ((((1-phinext[i,j]) * Dl + (1-phinext[i,j-1]) * Dl) / 2) * (c_l[i,j-1] - c_l[i,j]) / dx)
    
    #dc/dt completed term
    if phinext[i,j] < 1e-3 #if we are in liquid portion
      dcsdt = 0.0
      dcldt = (xplus_l + xmins_l + yplus_l + ymins_l) / dx
    elseif phinext[i,j] > 1-1e-3 #if we are in solid portion
      dcsdt = (xplus_s + xmins_s + yplus_s + ymins_s) / dx
      dcldt = 0.0
    else #if(phinext[i,j] >= 1e-9 && phinext[i,j] <= 1-1e-9) #if we are at the interface solve both equations
      dcsdt = ((xplus_s + xmins_s + yplus_s + ymins_s) / dx + P*phinext[i,j]*(1-phinext[i,j])*(mu_L - mu_S) + phinext[i,j] * dphidt_ij * (c_l[i,j] - c_s[i,j])) / phinext[i,j]
      dcldt = ((xplus_l + xmins_l + yplus_l + ymins_l) / dx + P*phinext[i,j]*(1-phinext[i,j])*(mu_S - mu_L) + (1-phinext[i,j]) * dphidt_opp * (c_s[i,j] - c_l[i,j])) / (1-phinext[i,j])
    end

    #update C
    @inbounds clnext[i,j] = c_l[i,j] + dt * dcldt
    @inbounds csnext[i,j] = c_s[i,j] + dt * dcsdt

    #limits of c_l and c_s
    if clnext[i,j] < 0.0
      clnext[i,j] = 0.0000001
    elseif clnext[i,j] > 1.0
      clnext[i,j] = 0.9999999
    end

    if csnext[i,j] < 0.0
      csnext[i,j] = 0.0000001
    elseif csnext[i,j] > 1.0
      csnext[i,j] = 0.9999999
    end


  end

  #---
  #---boundary conditions---
  #---

  # periodic conditions on the sides
  if i == 1 && j <= M
    clnext[1, j] = clnext[N-1, j]
    csnext[1, j] = csnext[N-1, j]
  end
  if i == N && j <= M
    clnext[N, j] = clnext[2, j]
    csnext[N, j] = csnext[2, j]
  end

  # no-flux conditions on the top and bottom
  if j == 1 && i <= N
    clnext[i, 1] = clnext[i, 2]
    csnext[i, 1] = csnext[i, 2]
  end

  if j == M && i <= N
    clnext[i, M] = clnext[i, M-1]
    csnext[i, M] = csnext[i, M-1]
  end

  return
end


function updateAll(phi::CuDeviceMatrix{Float64,1}, c_l::CuDeviceMatrix{Float64,1}, c_s::CuDeviceMatrix{Float64,1}, 
                  phinext::CuDeviceMatrix{Float64,1}, csnext::CuDeviceMatrix{Float64,1}, clnext::CuDeviceMatrix{Float64,1}, 
                  dt::Float64, N::Int64, M::Int64, kini::Float64, Vm::Float64, Tarr::CuDeviceVector{Float64,1}, R::Float64, dx::Float64,
                  G::Float64, Vs::Float64, t::Float64, Ts::Float64, xoffs::Float64)

  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

  if i <= N && j <= M && i >= 1 && j >= 1

    #copy over next values
    phi[i,j] = phinext[i,j]
    c_l[i,j] = clnext[i,j]
    c_s[i,j] = csnext[i,j]

  end

  return

end

function PullBack(phi::CuDeviceMatrix{Float64,1}, c_l::CuDeviceMatrix{Float64,1}, c_s::CuDeviceMatrix{Float64,1}, 
          N::Int64, M::Int64, c_0::Float64, kini::Float64, xoffs::Float64, dx::Float64)

  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

  if i <= N

    for j = 1:M

      if j < M

        phi[i, j] = phi[i, j+1]
        c_l[i, j] = c_l[i, j+1]
        c_s[i, j] = c_s[i, j+1]

      end

      if j == M

        #new values on boundary
        phi[i, M] = 0.0
        c_s[i,M] = kini*c_0
        c_l[i,M] = c_0

      end
    end
  end
  return
end

function TarrUpdate(Tarr::CuDeviceVector{Float64, 1}, M, Ts, G, dx, xoffs, Vs, t, initXpos)

  j = (blockIdx().x - 1) * blockDim().x + threadIdx().x

  if j <=M
  
    Tarr[j] = Ts + G*(j*dx - initXpos*dx + xoffs - Vs * t)

  end
  
  return
end

#GPU KERNAL - initialization of phi/circular seeds into domain
function IC_phi(phi::CuDeviceMatrix{Float64,1}, N::Int64, M::Int64, dx::Float64, posArr, xi)

  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

  #convert to meters
  x_p = j*dx

  if i <= N && j <= M && i >= 1 && j >= 1
    #in case of 2d
    xpos = posArr[i]

    #PHI Adding
    #add phi value of a single seed to the phi domain
    phi[i, j] = (1.0 .- tanh.((x_p - xpos) / (xi))) ./ 2

    #ensure no value of phi is greater than the limit, 1.0
    if phi[i, j] > 1.0
      phi[i, j] = 1.0
    end
  end

  #---
  #---boundary conditions---
  #---
  
  # periodic conditions on the sides
  if i == 1 && j <= M
    phi[1, j] = phi[N-1, j]
  end

  if i == N && j <= M
    phi[N, j] = phi[2, j]
  end

  # no flux conditions on the top and bottom
  if j == 1 && i <= N
    phi[i, 1] = phi[i, 2]
  end

  if j == M && i <= N
    phi[i, M] = phi[i, M-1]
  end
  return
end

#Represents the method for a single run of a benchmark problem starting at t=0 till end.
function single(filename::String, foldername::String, dx, dt, N, M, Vs, G, P, mu, timeEnd, initXpos, Ds, Dl, Tm, Ts)

  #Phase Field parameters
  xi = dx*4 #length of interface in meters
  sigma = 0.24 #J/m^2 interface energy

  #Constants
  R = 8.3145 #gas constant J/(mol*K)

  #Composition specific parameters
  Vm = 8.01e-6 #molar volume
  kini = 0.791 #initiated value of k
  c_0 = 0.17 #Composition of alloy

  me = -349 # K / wt % * 100 (slope of liquidus)

  #Anisotropy Parameters
  # noiseW = 0.1
  v_an = 0.018

  #GPU parameters
  nthreads = 16
  nblocksx = ceil(Int, N/nthreads)
  nblocksy = ceil(Int, M/nthreads)

  timeIteration = timeEnd/dt
  #Time values
  Tsave = collect(0:50000:timeIteration) #What values will the data be recorded at
  TsaveFull = collect(0:10000000:timeIteration) #These values describe when the entire PHI array will be recorded
  #define fields
  phi_h = fill(0.0, (N,M)) #host PHI array
  phi = CUDA.fill(0.0, (N,M)) #GPU PHI array

  c_s = CUDA.fill(kini*c_0, (N, M)) #GPU c_s array
  c_l = CUDA.fill(c_0, (N, M)) #GPU c_l array

  csnext = CUDA.fill(kini*c_0, (N, M))
  clnext = CUDA.fill(c_0, (N, M))

  Tarr_h = fill(Ts, M)
  Tarr = CUDA.fill(Ts, M) #GPU Temperature array

  xoffs = 0.0 #offset of moving frame from original location

  phinext = CUDA.fill(0.0, (N, M)) #next value of phi

  #Initial Conditions-----------------------------------
  #Phi and Tarr initial condition

  #Create initial condition in T
  for j = 1:M
    Tarr_h[j] = Ts + G*(j*dx - initXpos*dx)
  end

  copyto!(Tarr, Tarr_h)

  #Create initial condition in phi with hyperbolic tangent

  # Create perturbation and copy that array to GPU
  posArr_h = initXpos*dx .+ (rand(Float64, N) .- 0.5) .* dx #random ICs for perturbation
  #posArr_h = initXpos*dx -. 3*dx .+ [sin(π * x / N) for x in 0:N-1] .* 3*dx #sinusoidal ICs
  posArr = CUDA.fill(0.0, N)
  copyto!(posArr, posArr_h)
  #posArr = CUDA.fill(initXpos*2.5e-9, N) #for a 1d case or for no perturbation
  @cuda blocks = (nblocksx, nblocksy) threads=(nthreads, nthreads) IC_phi(phi, N, M, dx, posArr, xi)

  CUDA.seed!(1000) #Set the seed for the GPU random number generator

  # loop to march in time
  for iteration = 0:timeIteration

    t = round(iteration * dt, digits=25)

    beta = CUDA.rand(Float32, N, M)

    @cuda blocks=(nblocksx, nblocksy) threads=(nthreads, nthreads) solvePhi(phi, phinext, c_s, c_l,
                            N, M, Tarr, dx, R, Vm, dt, P, mu, xi, sigma, v_an, kini, me, Tm, beta)
    
    @cuda blocks=(nblocksx, nblocksy) threads=(nthreads, nthreads) SolveC(phinext, phi, c_s, c_l, csnext, clnext, Vm, Tarr, R, N, M, dx, dt, P, Ds, Dl, me, Tm, kini)
    @cuda blocks=(nblocksx, nblocksy) threads=(nthreads, nthreads) updateAll(phi, c_l, c_s, phinext, csnext, clnext, dt, N, M, kini, Vm, Tarr, R, dx, G, Vs, t, Ts, xoffs)

    #pullback method
    thresh = initXpos
    maxphi_thresh = findmax(phi[:, floor(Int64, thresh)])[1]
    if maxphi_thresh > 0.1
      @cuda blocks=(nblocksx) threads=(nthreads) PullBack(phi, c_l, c_s, N ,M, c_0, kini, xoffs, dx)
      xoffs = xoffs + dx
    end
    # println(xoffs)

    #Update Temperature
    @cuda blocks=(nblocksy) threads=(nthreads) TarrUpdate(Tarr, M, Ts, G, dx, xoffs, Vs, t, initXpos)

    if(iteration in Tsave)
      #calculate c from c_s and c_l
      c = c_s.*phi .+ c_l .* (1 .- phi)
      
      Vf = sum(phi[2:(N-1), 2:(M-1)]) / ((N-2)*(M-2)) #Find volume fraction
      Cf = sum(c[2:(N-1), 2:(M-1)]) / ((N-2)*(M-2)) #Find composition

      phi_Int = 0
      for i in 2:N-1
        phiinterface_ind = findmin(abs.(phi[i,:] .- 0.5))[2]
        if phiinterface_ind > phi_Int
          phi_Int = phiinterface_ind
        end
      end
      T_Int = Tarr[phi_Int]

      fileStream = open("$foldername/$filename", "a")
      write(fileStream, string(iteration, ",", Vf, ",", Cf, ",", T_Int, ",", xoffs, "\n"))
      close(fileStream)

      println("Iteration # == $iteration \t Volume Fraction (Phi) = $Vf \t Composition = $Cf \t Temp_Interface = $T_Int \t GrowDist = $(xoffs/dx)")

    end

    if(iteration in TsaveFull)
      # copy back to cpu
      #calculate c from c_s and c_l
      c = c_s .* phi .+ c_l .* (1 .- phi)

      copyto!(phi_h, phi)

      t_int = floor(Int64, iteration*dt*1e6)
      println("Saving file: iteration = $t_int")

      #This large number is used to help in the naming of the file
      phi_save = string("$foldername/phi_", t_int, ".csv")
      c_save = string("$foldername/c_$t_int.csv")

      open(c_save, "w") do io
        writedlm(io, c, ',')
      end

      open(phi_save, "w") do io
        writedlm(io, phi_h, ',')
      end

      # if(isnan(Vf) || isnan(Cf))
      #   return
      # end

    end
  end
  copyto!(Tarr_h, Tarr)

  return Tarr
end


println("GPU parallel relaxation model solve initiated")
dx = 4.0e-9 #m
Dl = 3.0e-9 #m^2 / s
Ds = 0.0 #m^2 / s
dt = 1.0e-11#dx^2 / (4*Dl) #s

Lengthx = 2000*dx# #meters
Lengthy = 3000*dx# #meters

N = floor(Int64, Lengthx / dx)
M = floor(Int64, Lengthy / dx)

P = 0.01 #permeability constant
mu = 3.15e-8 # interfacial mobility
Vs = 0.1 #solidification velocity (m/s)
G = 1e6 #temperature gradient (K/m)
initXpos = floor(Int64,M*0.8)
timeEnd = 0.0003 #seconds (when to stop the simulation)

Tm = 1811.0 #K
Ts = 1736.0 #K
Tarr = @time single("dendrite.csv", "P0_01/", dx, dt, N, M, Vs, G, P, mu, timeEnd, initXpos, Ds, Dl, Tm, Ts)
println("COMPLETED-----------------")
