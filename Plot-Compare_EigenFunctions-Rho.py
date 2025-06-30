# === First Code: Compare_EF_Rho ===
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import FortranFile as ff
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Times New Roman",
})
plt.rcParams.update({
    'font.size': 32,
    'legend.fontsize': 20,
    'axes.linewidth': 2.0,
    'lines.linewidth': 2.4,
    'lines.markersize': 14,
})
from scipy import stats
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import seaborn as sns
from matplotlib import cm
colors = sns.color_palette("viridis", 3)
markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', 'h', '+', 'x']

N = 2**10
delta = 2/5
nu = 0.1
omega = 0.05
b ,d =  0.315283 , 0.020692

folder = 'Data/IHBck-Fields-N-%d-delta-%.2f-nu-%.4f-omega-%.2f/'%(N,delta,nu,omega)
params = np.load(folder+'parameters.npz')
dtt = params['dtt']
tFinal = params['tFinal']
rhoEq = params['rhoEq']
L= params['Lx']
k = np.fft.fftfreq(N,1)*N*2*np.pi/L
x = np.arange(-np.pi, np.pi, 2*np.pi/N)
tVals = np.arange(0,tFinal,dtt)

gR_Analytic = np.loadtxt('gR_data.txt')
gI_Analytic = np.loadtxt('gI_data.txt')
g = (gR_Analytic + 1j * gI_Analytic)/ (d + 1j * b)
g = np.fft.ifftn(1j*k*np.fft.ifftn(rhoEq * g))
gR_Analytic = np.real(g)
gI_Analytic = np.imag(g)

Rx = np.sqrt(gR_Analytic**2 + gI_Analytic**2)
alphax = np.arctan2(gI_Analytic,gR_Analytic) 
beta = 2   
gFit = Rx*np.cos(alphax+beta)

ti= 1000
t = ti * dtt
vel = np.load(folder+'/rhoPerturb-t-%d.npy'%ti)

gFit = Rx*np.cos(alphax + beta)
def fit_function(x, a, beta):
    return a * Rx * np.cos(alphax + beta)

popt, pcov = curve_fit(fit_function, x, vel, p0=[1, beta])
fitted_a, fitted_beta = popt

colormap = cm.get_cmap('plasma', 10)
colors = [colormap(i) for i in range(10)]

plt.figure(figsize=(10, 9))
mf = matplotlib.ticker.ScalarFormatter(useMathText=True)
mf.set_powerlimits((-1,1))
plt.gca().yaxis.set_major_formatter(mf)
plt.text(0.05, 0.97, r'$\textbf{(b)}$', transform=plt.gca().transAxes,verticalalignment='top', horizontalalignment='center')
tStart = 1000 * dtt
for tidx, ti in enumerate(range(1000, 1200, 40)):
    t = ti * dtt
    vel = np.load(folder + '/rhoPerturb-t-%d.npy' % ti)
    gFit = fitted_a * Rx * np.cos(alphax + fitted_beta - b * (t - tStart)) * np.exp(-d * (t - tStart))
    plt.plot(x[0:-1:N // 35], vel[0:-1:N // 35], marker=markers[tidx % len(markers)], ls='none', alpha=0.6, mec='none',
             label=r'$%d$' % t )#color=colors[tidx % len(colors)])
    plt.plot(x, gFit, color='k', ls='--',alpha=0.8)
plt.plot(x, gFit, color='k', ls='--',alpha=0.8,label=r'${\rm Analytical}$')
plt.legend(title=r'$t=$',ncol=2, frameon=False)
plt.ylim([min(vel)-0.000035, 1.5*max(np.abs(vel))])
plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
# plt.xlim([-np.pi,np.pi])
plt.xlabel(r'$x$')
plt.ylabel(r'$\Delta \rho(x,t)$')
plt.tight_layout()
plt.savefig('Plots/Compare_EF_Rho.png', dpi=300, bbox_inches=None)
plt.show()
