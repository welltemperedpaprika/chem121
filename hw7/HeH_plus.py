# import packages for basic math, plotting, linear algebra, etc.
from numpy import *
from numpy.linalg import *
from numpy.random import *
from matplotlib.pyplot import *
from scipy.special import binom, erf, erfc

def eigSinvH(S,H):
    SinvH = inv(S) @ H
    E, U = eig(SinvH)

    order = argsort(real(E))
    c = zeros((K, K))
    for i in range(K):
        c[:, i] = real(U[:, order[i]])
        c[:, i] = c[:, i] / sqrt(c[:, i] @ S @ c[:, i])
    E = sort(real(E))
    return E, c

# Special function that helps to compute 2-e integrals.
def F0(x):
    if (x < 1e-8):
        return 1 - x/3
    else:
        return 0.5*sqrt(pi)*erf(sqrt(x))/sqrt(x)

# computes the overlap integral according to appendix of Szabo
def compute_overlap(alpha, beta, RA, RB):
    absum = alpha + beta
    abfac = alpha * beta / absum
    dRAB = RA - RB
    dRAB2 = dRAB @ dRAB
    return (pi/absum)**(3/2) * exp(-abfac * dRAB2)

# computes the kinetic integral according to appendix of Szabo
def compute_kinetic_energy(alpha, beta, RA, RB):
    absum = alpha + beta
    abfac = alpha * beta / absum
    dRAB = RA - RB
    dRAB2 = dRAB @ dRAB
    return (pi/absum)**(3/2) * exp(-abfac * dRAB2) * \
            abfac * (3 - 2*abfac * dRAB2)

# computes the attraction integral according to appendix of Szabo
def compute_elec_nuc_energy(alpha, beta, RA, RB, RC):
    absum = alpha + beta
    abfac = alpha * beta / absum
    dRAB = RA - RB
    dRAB2 = dRAB @ dRAB
    RP = (alpha*RA + beta*RB)/absum
    dRPC = RP - RC
    dRPC2 = dRPC @ dRPC
    return -(2*pi/absum)*exp(-abfac * dRAB2) * \
            F0(absum * dRPC2)

# computes the 2e- integral according to appendix of Szabo
def compute_elec_elec_energy(alpha,beta,gamma,delta,RA,RB,RC,RD):
    absum = alpha + beta
    abfac = alpha * beta / absum
    dRAB = RA - RB
    dRAB2 = dRAB @ dRAB
    RP = (alpha*RA + beta*RB)/absum

    gdsum = gamma + delta
    gdfac = gamma * delta / gdsum
    dRCD = RC - RD
    dRCD2 = dRCD @ dRCD
    RQ = (gamma*RC + delta*RD) / gdsum
    dRPQ = RP - RQ
    dRPQ2 = dRPQ @ dRPQ

    abgdsum = absum + gdsum
    abgdfac = absum * gdsum / abgdsum

    return 2 * pi**(5/2) * (absum * gdsum * sqrt(abgdsum))**(-1) * \
            exp(-abfac*dRAB2 - gdfac*dRCD2) * \
            F0(abgdfac * dRPQ2)

N = 2 # number of electrons
N_nuclei = 2
z_nuclei = array([2, 1]) # nuclear charges
K = 2 # number of basis functions
L = 3 # number of Gaussians used to build each basis function

widths = zeros((L,K)) # exponents of the gaussians
contraction_coeffs = zeros((L,K)) # we called these d
centers = zeros((L,K,3)) # position of the gaussians

# specific basis parameters
alpha1s_STO3G = array([0.109818, 0.405771, 2.22766])
d1s_STO3G = array([0.316894, 0.381531, 0.109991])

# 1s on He
zeta_He = 2.0925
widths[:,0] = alpha1s_STO3G * zeta_He**2
contraction_coeffs[:,0] = d1s_STO3G * widths[:,0]**(3/4)

# 1s on H
zeta_H = 1.24
widths[:,1] = alpha1s_STO3G * zeta_H**2
contraction_coeffs[:,1] = d1s_STO3G * widths[:,1]**(3/4)

# initialize nuclear array
R_nuclei = zeros((N_nuclei,3))
# R = 1.4632
# set internuclear distance
R = 100
R_nuclei[1,0] = R

centers[0,1,0] = R
centers[1,1,0] = R
centers[2,1,0] = R

# initialize matrices
# S:overlap, T: kinetic, U1: potential of nuclei1, U2: potential of nuclei2
S = zeros((K,K))
T = zeros((K,K))
U1 = zeros((K,K))
U2 = zeros((K,K,K,K))

# compute matrix elements
for mu in range(K):
    for A in range(L):
        alpha = widths[A,mu]
        dAmu = contraction_coeffs[A,mu]
        RA = centers[A,mu,:]

        for nu in range(K):
            for B in range(L):
                beta = widths[B,nu]
                dBnu = contraction_coeffs[B,nu]
                RB = centers[B,nu,:]

                S[mu,nu] += dAmu * dBnu * compute_overlap(alpha, beta, RA, RB)

                T[mu,nu] += dAmu * dBnu * compute_kinetic_energy(alpha, beta, RA, RB)

                for nucleus in range(N_nuclei):
                    U1[mu,nu] += dAmu * dBnu * z_nuclei[nucleus]  * \
                        compute_elec_nuc_energy(alpha, beta, RA, RB, R_nuclei[nucleus])

                for sigma in range(K):
                    for C in range(L):
                        gamma = widths[C,sigma]
                        dCsigma = contraction_coeffs[C,sigma]
                        RC = centers[C,sigma,:]

                        for lam in range(K):
                            for D in range(L):
                                delta = widths[D,lam]
                                dDlam = contraction_coeffs[D,lam]
                                RD = centers[D,lam,:]

                                U2[mu,nu,sigma,lam] += dAmu * dBnu * dCsigma * dDlam * \
                        compute_elec_elec_energy(alpha, beta, gamma, delta, RA, RB, RC, RD)

# initial guess to the scf procedure
h = T + U1
E, c = eigSinvH(S,h)

Nover2 = int(N/2)

# begin SCF, assumes convergence after 10 iterations
n_iterations = 10
for iterate in range(n_iterations):
    # make density matrix
    P = zeros((K,K))
    for mu in range(K):
        for nu in range(K):
            for j in range(Nover2):
                P[mu,nu] += 2 * c[mu,j] * c[nu,j]
    # make fock matrix
    F = copy(h)
    for mu in range(K):
        for nu in range(K):
            for lam in range(K):
                for sigma in range(K):
                    F[mu,nu] += P[lam,sigma] * \
                                ( U2[mu,nu,lam,sigma] - 0.5*U2[mu,sigma,lam,nu] )
    # make second guess
    E, c = eigSinvH(S,F)

    E_elec = sum(E[0:Nover2]) + 0.5 * trace(P @ h)
    print(E_elec)

# energy of nuclear repulsion
E_nuc = 0
for nucleus1 in range(N_nuclei):
    for nucleus2 in range(nucleus1+1,N_nuclei):
        dR = R_nuclei[nucleus1,:] - R_nuclei[nucleus2,:]
        dR2 = dR @ dR
        E_nuc += z_nuclei[nucleus1] * z_nuclei[nucleus2] / sqrt(dR2)

# total scf energy
E_total = E_elec + E_nuc
print(E_total)

# begin plotting
xvals = arange(-2,4,0.1)
yvals = arange(-3, 3, 0.1)
nvals = size(xvals)
state = 0
psi = zeros((nvals,nvals))
for i in range(nvals):
    x = xvals[i]
    for j in range(nvals):
        y = yvals[j]

        for mu in range(K):
            for A in range(L):
                alpha = widths[A,mu]
                dAmu = contraction_coeffs[A,mu]
                RA = centers[A,mu,:]
                xA = RA[0]
                yA = RA[1]
                phiA = exp(-alpha* ( (x-xA)**2 + (y-yA)**2 ) )

                psi[j,i] += c[mu,state] * dAmu * phiA

clf()
contourf(xvals, yvals, psi, levels=20, cmap=cm.seismic)
axis('equal')
text(0,0,'He',color='White',fontsize=14)
text(R,0,'H',color='White',fontsize=14)