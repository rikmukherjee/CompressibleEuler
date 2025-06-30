# =============================================================================
# 1D fluid in with a Trapping Potential
# Solved using Fourier Pseudo-Spectral Method
# =============================================================================
# 
# Governing Equations:
# 
#     \partial_t \rho + \partial_x (\rho u) = 0
#     \partial_t u   + \partial_x \left( \frac{1}{2} u^2 + D \rho^{\delta} + V(x) \right) = \nu\, \partial_x^2 u
#
#     where:
#         \rho(x,t)     = density field
#         u(x,t)        = velocity field
#         D             = equation of state coefficient
#         \delta        = polytropic exponent (e.g. 2/5, 1, 2)
#         V(x)          = external trapping potential (e.g. harmonic: V(x) = 0.5 * \omega^2 x^2)
#         \nu           = viscosity
#
#     Equation of state:
#         w = D \rho^{\delta}  (specific enthalpy form)
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

# --- Libraries ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import FortranFile as ff
from scipy import stats
from scipy.fft import rfft, irfft, rfftfreq
import os

# --- Matplotlib Configurations for LaTeX Text ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Times New Roman",
})
plt.rcParams.update({'font.size': 22})




#==============================================================================
# Parameter
#==============================================================================
N       = 2**10              # Number of spatial grid points
Lx      = 2*np.pi            # Domain size
tFinal  = 60                 # Final time
dt      = 0.001              # Time step
dtt     = 0.1                # Output interval
nu      = 0.1                # Viscosity coefficient
eta     = 0 * nu / 1000      # Density dissipation coefficient (zero here)
delta   = 2/5                # Exponent in isentropic EOS
dx      = Lx / N             # Spatial resolution
sigma   = Lx / 10            # Width of perturbation
x       = np.arange(-Lx/2, Lx/2, dx)  # Spatial grid
sigma_t = sigma              # Temporal width of perturbation (same as sigma)

#==============================================================================
# Steady state - Trap and Perturbation
#==============================================================================
omega       =   0.05 
trap        =   0.5*omega*x**2
trapfft     =   rfft(trap)
trapfft[N//8:] = 0
trapModif = irfft(trapfft)
D =1 
a = Lx/2 + dx
rhoEq = (a**2*omega/(2*D))**(1/delta) * (1-x**2/a**2)**(1/delta)
Mass        =   np.trapz(rhoEq, x)
rho0        =   rhoEq + 0.001*np.exp(-x**2/(2*sigma_t**2))
rho0        =   Mass*rho0/np.trapz(rho0, x)
plt.plot(x, rho0, label='Initial Condition')
plt.plot(x, rhoEq,'--k', label='Steady State')
plt.plot(x, trap, '--k', label='Trap')
plt.plot(x, trapModif, '--r', label='Modified Trap')
plt.show()

u0          =   0.0*np.sin(x)


#==============================================================================
k           =   rfftfreq(N, dx) * 2 * np.pi
rho         =   rfft(rho0)    
u           =   rfft(u0)
field       =   np.concatenate((rho, u))
kvec        =   np.concatenate((k, k))
semigroup   =   np.concatenate((np.exp(-eta * k**2 * dt), np.exp(-nu * k**2 * dt)))
viscTerm    =   np.concatenate((eta*k**2, nu*k**2))

#=======================================================================
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
    'omega': omega,
    'trap': trap,
    }

# Create folder if it doesn't exist: Name: 'Data/Fields-nu-%.4f-omega-%.4f' % (nu, omega)
folder = 'Data/FullNonlin-Fields-N-%d-delta-%.2f-nu-%.4f-omega-%.2f-small/'%(N,delta,nu,omega)
import os
if not os.path.exists(folder):
    os.makedirs(folder)

np.savez(os.path.join(folder, 'parameters.npz'), **parameters)  


def rhs(field, delta):
    # Input is in Fourier space
    rho = field[:len(k)]
    u   = field[len(k):]
    # Dealising 
    rho[N//3:] = 0
    u[N//3:] = 0
    # Compute the non-linear term
    nonlin1 = -1j * k * rfft(irfft(rho) * irfft(u))
    nonlin1[N//3:] = 0
    nonlin2 = -1j * k * rfft( (irfft(u))**2.0 / 2 + (np.abs(irfft(rho)))**delta  + trap)
    nonlin2[N//3:] = 0 
    nonlin = np.concatenate((nonlin1, nonlin2))
    return nonlin


def rhsRK4(field, delta):
    # input is in Fourier space but this involves dissipative term
    # Input is in Fourier space
    rho = field[:len(k)]
    u   = field[len(k):]
    # Compute the non-linear term
    nonlin1 = -1j * k * rfft(irfft(rho) * irfft(u))
    nonlin2 = -1j * k * rfft( (irfft(u))**2.0 / 2 + (irfft(rho))**delta)
    nonlin = np.concatenate((nonlin1, nonlin2))
    lin = np.concatenate((-nu*k**2*rho , -eta*k**2*u))
    return nonlin + lin



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

def RK4(field, nu, dt):
    # Input is in Fourier space
    k1 = rhsRK4(field, delta)
    k2 = rhsRK4(field + 0.5*dt*k1, delta)
    k3 = rhsRK4(field + 0.5*dt*k2, delta)
    k4 = rhsRK4(field + dt*k3, delta)
    field = field + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return field


np.savetxt('Data/PerturbIni.txt', rho0-rhoEq)
np.savetxt('Data/Trap.txt', trap)
np.savetxt('Data/SteadyState.txt', rhoEq)


for ti in range(int(tFinal/dtt)):
    # plt.clf()
    # plt.subplot(2,1,1)
    # plt.title(r'$t=%d$'%ti)
    # # plt.plot(x,rhoEq, 'k--',lw=2)
    # plt.plot(x, irfft(field[0:N//2+1])-rhoEq, '-',fillstyle='none')
    # plt.subplot(2,1,2)
    # plt.plot(x, irfft(field[N//2+1::]), '-',fillstyle='none')
    # plt.draw()
    # plt.pause(0.01)
    print('Time=%d/%d'%(ti,int(tFinal/dtt)))    
    np.save(os.path.join(folder, 'rho-t-%d.npy'%ti), irfft(field[0:N//2+1]))
    np.save(os.path.join(folder, 'u-t-%d.npy'%ti), irfft(field[N//2+1::]))


    for i in range(int(dtt // dt)):
        field  = IFEuler(field, semigroup, dt)
        #field = ExpEuler(field, nu, dt)
        # field = RK4(field, nu, dt)

plt.show()

