## =============================================================================
# Finite Difference code for compressible Euler equations
# in polar coordinates in 2 dimensional space.
# The boundary connditions are :
# u = 0 at r=0
# and d\rho/dr = 0 at r=0  
# At r=R we don't impose any boundary conditions since we evolve till the 
# profiles hit the boundary.
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
print("Starting simulation...")
t1=time.time()
import os

# Parameters
N  = 2**12
Lx = N
dx = 1
x = np.arange(0, Lx, dx)
dt = .1
delta = 2.0/5.0

sigma = 6
rho0 = 0.00000001+ np.exp(-x**2 / (2 * sigma**2))
u0   = 0*rho0
# rho0[0:N//2] = np.load('Data/N-1024-delta-1.00/Field-rho-99.npy')
# u0[0:N//2]   = np.load('Data/N-1024-delta-1.00/Field-u-99.npy')
nu =    2

def rhs(rho, u, delta):
    # Enforce BCs before derivative calculations
    rho[0] = rho[1]
    rho[-1] = rho[-2]
    u[0] = 0
    u[-1] = 0

    rhoVbyr = np.zeros_like(rho)
    with np.errstate(divide='ignore', invalid='ignore'):
        rhoVbyr[1:] = rho[1:] * u[1:] / x[1:]

    delu_dx = np.zeros_like(u)
    delu_dx[1:-1] = (u[2:] - u[:-2]) / (2*dx)
    delu_dx[0] = (u[1] - u[0]) / dx
    delu_dx[-1] = (u[-1] - u[-2]) / dx

    d2dx2u = np.zeros_like(u)
    d2dx2u[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    d2dx2u[0] = (u[1] - 2*u[0]) / dx**2
    d2dx2u[-1] = (u[-2] - 2*u[-1]) / dx**2

    delrho_dx = np.zeros_like(rho)
    delrho_dx[1:-1] = (rho[2:] - rho[:-2]) / (2*dx)
    delrho_dx[0] = (rho[1] - rho[0]) / dx
    delrho_dx[-1] = (rho[-1] - rho[-2]) / dx

    d2dx2rho = np.zeros_like(rho)
    d2dx2rho[1:-1] = (rho[2:] - 2*rho[1:-1] + rho[:-2]) / dx**2
    d2dx2rho[0] = (rho[1] - 2*rho[0] + rho[1]) / dx**2
    d2dx2rho[-1] = (rho[-2] - 2*rho[-1] + rho[-2]) / dx**2

    rhsrho = -u * delrho_dx - rho * delu_dx - rhoVbyr  + nu * d2dx2rho #(if diffusion added)
    rhsu = -u * delu_dx - delta * rho**(delta - 1) * delrho_dx + nu * d2dx2u

    # Apply BCs on RHS to ensure u stays zero at boundaries
    rhsu[0] = 0
    rhsu[-1] = 0

    return rhsrho, rhsu


rho = rho0.copy()
u   = u0.copy()

def Euler(rho,u,nu,dt,delta):
    # Input is in Real space
    rhsrho,rhsu = rhs(rho,u,delta)
    rho = rho + rhsrho*dt
    u = u + rhsu*dt
    return rho,u

def CranckNicolson(rho,u,nu,dt,delta):
    # Input is in Real space
    rhsrho,rhsu = rhs(rho,u,delta)
    rho = rho + 0.5*(rhsrho + rhs(rho,u,delta)[0])*dt
    u = u + 0.5*(rhsu + rhs(rho,u,delta)[1])*dt
    return rho,u

t = 0
dtt = 100
tFinal = 10000   
folder = 'N-%d-delta-%.2f/'%(N,delta)
try:
    os.makedirs('Data/'+folder)
except:
    pass

np.savez('Data/'+folder+'Parameters.npz',N=N,nu=nu,delta=delta,tFinal=tFinal,dtt=dtt)


for ti in range(int(tFinal/dtt)):
    # if(ti==10):dt=dt*2
    # if(ti==10):nu=nu/2
    plt.clf()
    plt.subplot(1,2,1)
    plt.plot(x,rho)
    plt.subplot(1,2,2)
    plt.plot(x,u,'--')
    plt.draw()
    plt.pause(0.01)
    print("Time = %d/%d"%(ti,tFinal//dtt))
    np.save('Data/'+folder+'Field-rho-%d.npy'%ti,rho)
    np.save('Data/'+folder+'Field-u-%d.npy'%ti,u)
    for i in range(int(dtt/dt)):
        rho,u = Euler(rho,u,nu,dt,delta)
        # rho,u = CranckNicolson(rho,u,nu,dt,delta)

t2=time.time()  
print("Simulation ended in ",t2-t1," seconds")

plt.show()
# =============================================================================




# def rhs(rho,u,delta):
#     # Boundary condition for rho : derivative at the boundary is zero
#     rho[0] = rho[1]
#     rho[-1] = rho[-2]
#     # Boundary condition for u : u = 0 at the boundary
#     u[0] = 0
#     u[-1] = 0

#     # Compute the non-linear terms 
#     rhoVbyr = 0*rho
#     rhoVbyr[0] = rho[1]*u[0]/x[1]
#     rhoVbyr[1::] = rho[1::]*u[1::]/x[1::]

#     # Here u=0 at the boundary
#     delu_dx = 0*u
#     delu_dx[0] = u[1]/x[1]
#     delu_dx[1:-1] = (u[2::]-u[0:-2])/dx
#     delu_dx[-1] = u[-2]/x[-2]

#     # Compute d2dx2 u enuring u=0 at the boundary
#     d2dx2u = 0*u
#     d2dx2u[0] = (u[1]-2*u[0])/dx**2
#     d2dx2u[1:-1] = (u[2::]-2*u[1:-1]+u[0:-2])/dx**2
#     d2dx2u[-1] = (u[-2]-2*u[-1])/dx**2
    
#     # Here delrho_dx = 0 at the boundary
#     delrho_dx = 0*rho
#     delrho_dx[0] = (rho[1]-rho[0])/dx
#     delrho_dx[1:-1] = (rho[2::]-rho[0:-2])/dx
#     delrho_dx[-1] = (rho[-2]-rho[-1])/dx

#     # Compute d2dx2 rho ensuring delrho_dx=0 at the boundary
#     d2dx2rho = 0*rho
#     d2dx2rho[0] = (rho[1]-2*rho[0])/dx**2
#     d2dx2rho[1:-1] = (rho[2::]-2*rho[1:-1]+rho[0:-2])/dx**2
#     d2dx2rho[-1] = (rho[-2]-2*rho[-1])/dx**2




#     rhsrho = -u*delrho_dx - rho*delu_dx - rhoVbyr  #+ nu*d2dx2rho

#     rhsu = -u*delu_dx - delta*rho**(delta-1)*delrho_dx +  nu*d2dx2u

#     return rhsrho,rhsu
