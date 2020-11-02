# import packages for basic math, plotting, linear algebra, etc.
from numpy import *
from numpy.linalg import *
from numpy.random import *
from matplotlib.pyplot import *
from scipy.special import binom, erf, erfc

def eigSinvH(S,H):
    SinvH = inv(S) @ H
    E, U = eig(SinvH)

    order = argsort(E)
    c = zeros((K, K))
    for i in range(K):
        c[:, i] = U[:, order[i]]
        c[:, i] = c[:, i] / sqrt(c[:, i] @ S @ c[:, i])

    E = sort(E)

    return E, c


alpha = 2
deltax = 0.5

n = 10
K = 2*n + 1

center = arange(-n*deltax,(n+1)*deltax,deltax)

S = zeros((K,K))
h = zeros((K,K))
G = zeros((K,K))

#  calculate matrix elements
for A in range(K):
    xA = center[A]
    for B in range(K):
        xB = center[B]

        S[A,B] = sqrt(0.5*pi/alpha) * exp(-0.5*alpha* (xA - xB)**2 )

        h[A,B] = 0.5*S[A,B] * (alpha - alpha**2 * (xA - xB)**2 + \
                             0.25*(1/alpha + (xA + xB)**2 ))

        G[A,B] = S[A,B] * ( 3/(16*alpha**2) + \
                            (3/(8*alpha)) * (xA + xB)**2 + \
                            (1/16) * (xA + xB)**4  )

E, c = eigSinvH(S,h)
c1 = c[:,0]
c2 = c[:,0]

# interaction strength
a = 1

# begin scf procedure
niterations = 100
for iteration in range(niterations):
    avx4 = c1 @ G @ c1
    avy4 = c2 @ G @ c2

    heffx = h + a * avy4 * G
    heffy = h + a * avx4 * G

    E, c = eigSinvH(S,heffx)
    c1 = c[:,0]
    e1 = E[0]

    E, c = eigSinvH(S,heffy)
    c2 = c[:,0]
    e2 = E[0]

    Etot = e1 + e2 - a * avx4 * avy4
print('mean field energy = ',Etot)

# begin CI calculation
N_orb = 16  # number of orbitals
N_config = N_orb**2  # number of configurations possible, ignoring spin

config_list = zeros((N_config,2)).astype(int)  # initialize possible configs as [i,j]th occupied orbitals
count = 0
for i in range(N_orb):
    for j in range(N_orb):
        config_list[count,:] = [i,j]
        count += 1

# initialize CI matrix
H_CI = zeros((N_config,N_config))

# populate CI matrix elements \bra \psi_ij | H | \psi_ij \ket, where \psi is a configuration
for config1 in range(N_config):
    i, j = config_list[config1,:]
    for config2 in range(N_config):
        k, l = config_list[config2,:]

        if j==l:
            H_CI[config1,config2] += c[:,i] @ h @ c[:,k]
        if i==k:
            H_CI[config1,config2] += c[:,j] @ h @ c[:,l]

        H_CI[config1, config2] += a * (c[:,k] @ G @ c[:,i]) * (c[:,l] @ G @ c[:,j])

E, b = eig(H_CI)
order = argsort(E)
E = real(sort(E))
print(E[0])
# Exact energy for a=1 is about 1.14

# plotting the solutions
xvals = arange(-2.5,2.5,0.1)
yvals = xvals
nvals = size(xvals)

psi = zeros((nvals,nvals))

# for each configuration, add up the CI solution wavefunctions to get a improved solution from just using one config.
from numba import jit
@jit(nopython=True)
def compute_wavefunction(x,y):
    psixy = 0
    for config in range(N_config):
        i, j = config_list[config, :]
        for A in range(K):
            xA = center[A]
            phiA = exp(-alpha * (x - xA) ** 2)
            for B in range(K):
                yB = center[B]
                phiB = exp(-alpha * (y - yB) ** 2)
                psixy += b[config, order[0]] * c[A, i] * c[B, j] * phiA * phiB

    return real(psixy)

for xindex in range(nvals):
    x = xvals[xindex]
    for yindex in range(nvals):
        y = yvals[yindex]

        psi[xindex, yindex] = compute_wavefunction(x,y)

clf()
contourf(xvals, yvals, psi, levels=20, cmap=cm.seismic)
axis('equal')



