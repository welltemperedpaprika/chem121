# import packages for basic math, plotting, linear algebra, etc.
from numpy import *
from numpy.linalg import *
from matplotlib.pyplot import *

alpha = 2
deltax = 0.5

nmax = 20

Kvals = zeros(nmax+1)
Evals = zeros((nmax+1,9))

for n in range(4,nmax+1):
    K = 2*n + 1

    center = arange(-n*deltax,(n+1)*deltax,deltax)

    S = zeros((K,K))
    H = zeros((K,K))

    for A in range(K):
        xA = center[A]
        for B in range(K):
            xB = center[B]

            S[A,B] = sqrt(0.5*pi/alpha) * exp(-0.5*alpha* (xA - xB)**2 )

            H[A,B] = 0.5*S[A,B] * (alpha - alpha**2 * (xA - xB)**2 + \
                                 0.25*(1/alpha + (xA + xB)**2 ))

    SinvH = inv(S) @ H
    E, U = eig(SinvH)

    order = argsort(E)
    c = zeros((K,K))
    for i in range(K):
        c[:,i] = U[:,order[i]]
        c[:,i] = c[:,i] / sqrt(c[:,i] @ S @ c[:,i])

    E = sort(E)

    Kvals[n] = K
    Evals[n,:] = E[0:9]

clf()
for i in range(9):
    plot(Kvals,Evals[:,i])

xlim(9,2*nmax+1)
xlabel('# of basis functions',fontsize=14)
ylabel('energy',fontsize=14)
show()