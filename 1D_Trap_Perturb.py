# =============================================================================
# We solve the perturbation equations for the compressible Euler equations
# =============================================================================
# Let, rhoEq be the steady-state density profile:
#
#     \rho_eq(x) = ( \frac{\omega a^2}{2D} )^{1/\delta} (1 - \frac{x^2}{a^2} )^{1/\delta}
#
# where:
#     - a is constant related to the trap size
#     - \omega is the frequency of the harmonic trap
#     - \delta is the exponent in the equation of state
#
# The linearized perturbation equations around this steady state are:
#
#     \partial_t \rho + \partial_x (\rho_eq  u) = 0
#     \partial_t u + D \delta \partial_x  \rho_eq^{\delta - 1} \rho ) = \nu \partial_x^2 u
#
# Method:
#     - Fourier pseudo-spectral method in space
#     - Nonlinear terms computed in real space and transformed with FFT
#     - Dealiasing applied via 2/3-rule
#     - Time integration using:
#         - Integrating Factor Euler (IFEuler)
#         - Exponential Euler
#         - Runge-Kutta 4th order (RK4)
#
# Initial condition:
#     - \rho(x,0): small perturbation around a steady equilibrium
#     - u(x,0) = 0
#
# Outputs:
#     - \rho(x,t) and u(x,t) saved at intervals
#     - Steady state, trap, and perturbation stored separately
# =============================================================================
# Outputs
# =============================================================================
# - Saves \rho(x,t) and u(x,t) at regular time intervals as .npy files.
# - Also saves initial perturbation, trap profile, and equilibrium state.
# =============================================================================


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import FortranFile as ff
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Times New Roman",
})
plt.rcParams.update({'font.size': 22})
from scipy import stats
from scipy.fft import rfft, irfft, rfftfreq


#==============================================================================
# Parameter
#==============================================================================
# =============================================================================
# Parameter definitions
# =============================================================================
N           =   2**10          # Number of spatial grid points
Lx          =   2*np.pi        # Domain length
tFinal      =   300            # Final simulation time
dt          =   0.001          # Time step size
dtt         =   0.1            # Output interval (coarse time steps)
nu          =   0.06           # Viscosity coefficient (acts on velocity field)
eta         =   0*nu/100       # Artificial viscosity on density (optional, here zero)
delta       =   2/5            # Polytropic exponent in equation of state
dx          =   Lx/N           # Grid spacing
x           =   np.arange(-Lx/2, Lx/2, dx)  # Spatial grid (centered around 0)

# =============================================================================
# Steady-state profile and trapping potential
# =============================================================================
omega       =   0.05                     # Frequency of the harmonic trapping potential
trap        =   0.5 * omega * x**2       # Harmonic trap: V(x) = (1/2) * omega * x^2

# Analytical steady-state density profile:
#     rho_eq(x) = [ (omega * a^2)/(2D) ]^{1/delta} * (1 - x^2/a^2)^{1/delta}
# where 'a' is chosen to make the support just inside the domain [-Lx/2, Lx/2]
D           =   1                        # Equation of state constant
a           =   Lx/2 + dx                # Maximum extent of the trap support
rhoEq       =   (a**2 * omega / (2 * D))**(1/delta) * (1 - x**2 / a**2)**(1/delta)

# =============================================================================
# Initial perturbation 
# =============================================================================
rho0        =   0.0 * np.sin(x)          # Initial perturbation to density (here zero)
u0          =   0.001 * np.sin(x)        # Small sinusoidal perturbation to velocity

# =============================================================================
# Plotting:
# =============================================================================
plt.plot(x, rhoEq, label='Steady State $\rho_{\mathrm{eq}}(x)$')
plt.plot(x, trap, '--k', label='Trap $V(x)$')
plt.plot(x, rho0, '--k', label='Initial Perturbation')
plt.legend()
plt.show()

# ==============================================================================
# save the parameters in a file
parameters = {
    'N': N,
    'Lx': Lx,
    'tFinal': tFinal,
    'dt': dt,
    'dtt': dtt,
    'nu': nu,
    'eta': eta,
    'delta': delta,
    'dx': dx,
    'rhoEq': rhoEq,
    'omega': omega
    }

# Create folder if it doesn't exist: Name: 'Data/Fields-nu-%.4f-omega-%.4f' % (nu, omega)
folder = 'Data/IHBck-Fields-N-%d-delta-%.2f-nu-%.4f-omega-%.2f/'%(N,delta,nu,omega)
import os
if not os.path.exists(folder):
    os.makedirs(folder)

np.savez(os.path.join(folder, 'parameters.npz'), **parameters)    


#==============================================================================
k           =   rfftfreq(N, dx) * 2 * np.pi
rho         =   rfft(rho0)    
u           =   rfft(u0)
field       =   np.concatenate((rho, u))
kvec        =   np.concatenate((k, k))
viscTerm    =   np.concatenate((eta*k**2, nu*k**2))
semigroup   =   np.concatenate((np.exp(-eta * k**2 * dt), np.exp(-nu * k**2 * dt)))


def rhs(field, delta):
    # Input is in Fourier space
    rho = field[:len(k)]
    u   = field[len(k):]
    # Compute the non-linear term
    nonlin1 = -        1j * k * rfft(  rhoEq * irfft(u))
    nonlin2 = - delta* 1j * k * rfft(  rhoEq**(delta-1) * irfft(rho) )
    nonlin = np.concatenate((nonlin1, nonlin2))
    return nonlin



def IFEuler(field, semigroup,dt):
    '''
    This function Integrating factor
    Euler time stepping where the liner 
    term is integrated exactly.
    '''
    field = semigroup * (field + rhs(field, delta) * dt)
    return field

def ExpEuler(field, nu, dt):
    '''
    This is exponential Euler time stepping
    where the linear term is integrated
    exactly. Slightly different from IFEuler
    This integration does not work for zero 
    viscosity.
    '''
    field = field * semigroup + (rhs(field,delta) / (viscTerm + 1e-15)) * (1 - semigroup)
    return field

# def RK4(field, nu, dt):
#     # Input is in Fourier space
#     k1 = rhsRK4(field, delta)
#     k2 = rhsRK4(field + 0.5*dt*k1, delta)
#     k3 = rhsRK4(field + 0.5*dt*k2, delta)
#     k4 = rhsRK4(field + dt*k3, delta)
#     field = field + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
#     return field





for ti in range(int(tFinal/dtt)):
    print('Time=%d/%d'%(ti,int(tFinal/dtt)))
    # plt.clf()
    # plt.plot(x, irfft(field[0:N//2+1]), '-',fillstyle='none')
    # plt.plot(x, irfft(field[N//2+1::]), '-',fillstyle='none')
    # # plt.plot(x,rhoEq, 'k--',lw=2)
    # plt.title(f"Time: {ti*dtt}")
    # plt.draw()
    # plt.pause(0.01)
    np.save(os.path.join(folder, 'rhoPerturb-t-%d.npy'%ti), irfft(field[0:N//2+1]))
    np.save(os.path.join(folder, 'uPerturb-t-%d.npy'%ti), irfft(field[N//2+1::]))
    # np.save('Data/rhoPerturb-t-%d.npy'%ti, irfft(field[0:N//2+1]))
    # np.save('Data/uPerturb-t-%d.npy'%ti, irfft(field[N//2+1::]))

    for i in range(int(dtt // dt)):
        field  = IFEuler(field, semigroup, dt)
        #field = ExpEuler(field, nu, dt)
        # field = RK4(field, nu, dt)

plt.show()
