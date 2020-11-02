# import packages for basic math, plotting, linear algebra, random number generation, etc.from numpy import *
from numpy import *
from numpy.linalg import *
from numpy.random import *
from matplotlib.pyplot import *
from scipy.special import binom, erf, erfc

#trial wavefunction basis parameter
alpha = 2
deltax = 0.5

#separation of each gaussian basis
n = 10
#number of basis
K = 2*n + 1

#the center of each basis
center = arange(-n*deltax,(n+1)*deltax,deltax)

#initialize overlap matrix and hamiltonian matrix
S = zeros((K,K))
H = zeros((K,K))

#iterate over the matrices while allocating each matrix element
for A in range(K):
    xA = center[A]
    for B in range(K):
        xB = center[B]

        S[A,B] = sqrt(0.5*pi/alpha) * exp(-0.5*alpha* (xA - xB)**2 )

        H[A,B] = 0.5*S[A,B] * (alpha - alpha**2 * (xA - xB)**2 + \
                             0.25*(1/alpha + (xA + xB)**2 ))

#Solve the eigenvalue equation Hx = ESx
SinvH = inv(S) @ H
#E contains the energy values on the diagonal,
# and U contains the coefficients of the linear combination with the basis functions
E, U = eig(SinvH)

#order the energies from small to largest and the corresponding
# eigenvectors (linear combination of trial wavefunctions) accordingly and store them in c
# renormalize each coeff vector, such that /sum |c^'_i|^2*Sii = 1, where c^'_i is the
# normalized coeff vector we want
order = argsort(E)
c = zeros((K,K))
for i in range(K):
    c[:,i] = U[:,order[i]]
    c[:,i] = c[:,i] / sqrt(c[:,i] @ S @ c[:,i])
E = sort(E)
print(E)

cla()
xvals = arange(-4,4,0.01)  # set values of psi to be evaluated on
psi = 0 * xvals
# from our basis functions, construct a linear combination using coeffs from c
for A in range(K):
    phiA = exp(-alpha * (xvals - center[A])**2 )
    psi = psi + c[A,0]*phiA

plot(xvals,psi,linewidth=4)
show()
psi0exact = pi**(-0.25) * exp(-0.5* xvals**2)
plot(xvals,psi0exact)
show()