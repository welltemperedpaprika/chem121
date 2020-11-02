# import packages for basic math, plotting, linear algebra, etc.
from numpy import *
from numpy.linalg import *
from numpy.random import *
from matplotlib.pyplot import *
from scipy.special import binom, erf, erfc

# Solves the generalized eigenvalue problem, with eigenvectors normalized s.t c.T @ S @ c = 1 on the diagonal.
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

# set width and spacing of our gaussian bases.
alpha = 2
deltax = 0.5

# set number of centers and basis functions.
n = 10
K = 2*n + 1

center = arange(-n*deltax,(n+1)*deltax,deltax)

# initialize relevant matrices.
S = zeros((K,K))
h = zeros((K,K))
G = zeros((K,K))

# allocate matrix elements given by the computed integral's form.
for A in range(K):
    xA = center[A]
    for B in range(K):
        xB = center[B]

        S[A,B] = sqrt(0.5*pi/alpha) * exp(-0.5*alpha* (xA - xB)**2 )

        h[A,B] = 0.5*S[A,B] * (alpha - alpha**2 * (xA - xB)**2 +
                             0.25*(1/alpha + (xA + xB)**2 ))

        G[A,B] = S[A,B] * ( 3/(16*alpha**2) +
                            (3/(8*alpha)) * (xA + xB)**2 +
                            (1/16) * (xA + xB)**4  )

# solve the eigenvalue problem.
E, c = eigSinvH(S,h)
c1 = c[:,0]  # gnd state coeff of first oscillator
c2 = c[:,2]  # gnd state coeff of second oscillator

a = 1 # interaction strength

# begin SCF procedure. Assumes convergence after 100 iterations.
niterations = 100
for iteration in range(niterations):
    # computes the effective mean field potential
    avx4 = c1 @ G @ c1
    avy4 = c2 @ G @ c2

    # computes the effective hamitonians
    heffx = h + a * avy4 * G
    heffy = h + a * avx4 * G

    # solves the eigenvalue equation for oscillator 1
    E, c = eigSinvH(S,heffx)
    c1 = c[:,0]
    e1 = E[0]

    # solves the eigenvalue equation for oscillator 2
    E, c = eigSinvH(S,heffy)
    c2 = c[:,0]
    e2 = E[0]

    # ground state energy
    Etot = e1 + e2 - a * avx4 * avy4
    print(Etot)

# Exact energy for a=1 is about 1.14

# initialize values to query for wavefunction
xvals = arange(-2.5,2.5,0.1)
yvals = xvals
nvals = size(xvals)

chi1 = 0 * xvals
chi2 = 0 * yvals

# computes wavefunction for each oscillator
for A in range(K):
    chi1 += c1[A] * exp(-alpha * (xvals - center[A])**2 )
    chi2 += c2[A] * exp(-alpha * (xvals - center[A])**2 )

# plot(xvals,chi1,'o')
# plot(yvals,chi2)

# begin plotting

clf()
psi = zeros((nvals,nvals))

# computes total wavefunction
for i in range(nvals):
    x = xvals[i]
    for j in range(nvals):
        y = yvals[j]

        psi[i,j] = chi1[i] * chi2[j]

clf()
contourf(xvals, yvals, psi, levels=20, cmap=cm.seismic)
axis('equal')

