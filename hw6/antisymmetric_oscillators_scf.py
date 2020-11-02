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

for A in range(K):
    xA = center[A]
    for B in range(K):
        xB = center[B]
        #  populate overlap matrix element
        S[A,B] = sqrt(0.5*pi/alpha) * exp(-0.5*alpha* (xA - xB)**2 )
        #  populate hamiltonian matrix element
        h[A,B] = 0.5*S[A,B] * (alpha - alpha**2 * (xA - xB)**2 + \
                             0.25*(1/alpha + (xA + xB)**2 ))
        #  populate g matrix element
        G[A,B] = S[A,B] * ( 3/(16*alpha**2) + \
                            (3/(8*alpha)) * (xA + xB)**2 + \
                            (1/16) * (xA + xB)**4  )

# initial guess for the eigenvalue and eigenvector
E, c = eigSinvH(S,h)
c1 = c[:,0]
c2 = c[:,1]

a = 1
#  initial guess for the density matrix
P = zeros((K,K))
#  set limit of SCF iterations, assume to converge
niterations = 100
for iteration in range(niterations):

    for D in range(K):
        for E in range(K):
            P[D,E] = c2[D]*c2[E] + c1[D]*c1[E]
    # P = outer(c1,c1) + outer(c2,c2)
    #  construct the "fock" matrix
    heff = h + a * ( trace(P @ G)*G - G @ P @ G )
    # update the eigenvalue and eigenvectors
    E, c = eigSinvH(S,heff)
    c1 = c[:,0]
    c2 = c[:,1]
    e1 = E[0]
    e2 = E[1]
    # prints the total energy of the system
    Etot = 0.5 * ( e1 + e2 + trace(P @ h) )
    print(Etot)

# Exact energy for a=1 is about 2.438

# set plotting parameters
xvals = arange(-2.5,2.5,0.1)
yvals = xvals
nvals = size(xvals)

chi1 = 0 * xvals
chi2 = 0 * yvals

# begin allocating the individual wavefunctions for each oscillator
for A in range(K):
    chi1 += c1[A] * exp(-alpha * (xvals - center[A])**2 )
    chi2 += c2[A] * exp(-alpha * (xvals - center[A])**2 )
plot(xvals,chi1)
plot(yvals,chi2)

clf()
psi = zeros((nvals,nvals))
# calculate the total wavefunction of the combined system.
for i in range(nvals):
    x = xvals[i]
    for j in range(nvals):
        y = yvals[j]

        psi[i,j] = (1/sqrt(2))*(chi1[i] * chi2[j] \
                                - chi2[i] * chi1[j] )

clf()
contourf(xvals, yvals, psi, levels=20, cmap=cm.seismic)
axis('equal')

