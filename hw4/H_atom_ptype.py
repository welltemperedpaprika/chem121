# import packages for basic math, plotting, linear algebra, etc.
from numpy import *
from numpy.linalg import *
from numpy.random import *
from matplotlib.pyplot import *
from scipy.special import binom, erf, erfc

def eigSinvH(S,H):
    """
    Solves the eigenvalue equation Hc=SEc.
    Parameters
    ----------
    S  2d array, the overlap matrix
    H  2d array, the hamiltonian matrix

    Returns
    -------
    E  array, the energy eigenvalues, sorted
    c  2d array, the coefficient matrix, sorted according to E
    """
    SinvH = inv(S) @ H
    E, U = eig(SinvH)

    order = argsort(real(E))
    c = zeros((K, K))
    for i in range(K):
        c[:, i] = real(U[:, order[i]])
        c[:, i] = c[:, i] / sqrt(c[:, i] @ S @ c[:, i])
    E = sort(real(E))
    return E, c

def F0(x):
    """
    Computes the F0 part of the nuclear term of the hamiltonian
    Parameters
    ----------
    x  float

    Returns
    -------
    float
    """
    if (x < 1e-8):
        return 1 - x/3
    else:
        return 0.5*sqrt(pi)*erf(sqrt(x))/sqrt(x)

def compute_overlap(alpha, beta, RA, RB):
    """
    Computes the overlap matrix element given by the formula in Appendix A of Szabo & Ostlund.
    Parameters
    ----------
    alpha  float, width of primitive gaussian a
    beta  float, width of primitive gaussian b
    RA  array, position of electron A
    RB  array, position of electron B

    Returns
    -------
    float
    """
    absum = alpha + beta
    abfac = alpha * beta / absum
    dRAB = RA - RB
    dRAB2 = dRAB @ dRAB
    return (pi/absum)**(3/2) * exp(-abfac * dRAB2)

def compute_kinetic_energy(alpha, beta, RA, RB):
    """
    Computes the kinetic energy matrix element given by the formula in Appendix A of Szabo & Ostlund
    Parameters
    ----------
    alpha  float, width of primitive gaussian a
    beta  float, width of primitive gaussian b
    RA  array, position of electron A
    RB  array, position of electron B

    Returns
    -------
    float
    """
    absum = alpha + beta
    abfac = alpha * beta / absum
    dRAB = RA - RB
    dRAB2 = dRAB @ dRAB
    return (pi/absum)**(3/2) * exp(-abfac * dRAB2) * \
            abfac * (3 - 2*abfac * dRAB2)

def compute_elec_nuc_energy(alpha, beta, RA, RB, RC):
    """
    Computes the electron-nuclear attraction matrix element given by the formula in appendix A of Szabo and Ostlund
    Parameters
    ----------
    alpha  float, width of primitive gaussian a
    beta  float, width of primitive gaussian b
    RA  array, position of electron A
    RB  array, position of electron B
    RC  array, position of nucleus

    Returns
    -------

    """
    absum = alpha + beta
    abfac = alpha * beta / absum
    dRAB = RA - RB
    dRAB2 = dRAB @ dRAB
    RP = (alpha*RA + beta*RB)/absum
    dRPC = RP - RC
    dRPC2 = dRPC @ dRPC
    return -(2*pi/absum)*exp(-abfac * dRAB2) * \
            F0(absum * dRPC2)

K = 3 # total basis function number
L = 2 # number of Gaussians used to build each basis function

# 2d array of width parameter. widths[i][j] gives the width of ith primitive gaussian of the
# jth basis function
widths = zeros((L,K))
contraction_coeffs = zeros((L,K)) # we called these d
centers = zeros((L,K,3))  # initialize the 3d array corresponding to the xyz coord of each primitive gaussian.

# Set width params
widths[:,0] = [0.6, 0.6]
widths[:,1] = [0.1, 0.1]
widths[:,2] = [0.02, 0.02]

# Set contraction coeffs
contraction_coeffs[:,0] = [1, -1]
contraction_coeffs[:,1] = [1, -1]
contraction_coeffs[:,2] = [1, -1]

offset = 0.01 # we called this Delta
# Set the distance of each basis function away from the origin
centers[:,0,2] = [offset, -offset]
centers[:,1,2] = [offset, -offset]
centers[:,2,2] = [offset, -offset]

# Let the coord of the nucleus be 0,0,0
R_nucleus = zeros(3)

# Initialize overlap, kinetic, and nuclear matrix
S = zeros((K,K))
T = zeros((K,K))
U = zeros((K,K))

# Populate the matrix element using previous relevant functions into the corresponding matrices.
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
                # we have to take contac. coeff. into account because our basis func. is made of linear comb.
                # of primitive gaussians now.
                S[mu,nu] += dAmu * dBnu * compute_overlap(alpha, beta, RA, RB)

                T[mu,nu] += dAmu * dBnu * compute_kinetic_energy(alpha, beta, RA, RB)

                U[mu,nu] += dAmu * dBnu * compute_elec_nuc_energy(alpha, beta, RA, RB, R_nucleus)

H = T + U
E, c = eigSinvH(S,H)  # Solves the eigenvalue equation

print(E)

# Initialize relevant points to query for plotting purposes.
x_values = arange(-10,10,0.1)
z_values = x_values
number_of_values = size(x_values)
# Plot the ground state wavefunction
state = 0
psi = zeros((number_of_values,number_of_values))

# Calculates the linear combination of basis functions given the coefficient matrix.
for i in range(number_of_values):
    x = x_values[i]
    for j in range(number_of_values):
        z = z_values[j]

        for mu in range(K):
            for A in range(L):
                alpha = widths[A, mu]
                dAmu = contraction_coeffs[A, mu]
                RA = centers[A, mu, :]
                xA = RA[0]
                zA = RA[2]
                phiA = exp(-alpha * ( (x-xA)**2 + (z-zA)**2  ) )

                psi[i,j] += c[mu,state] * dAmu * phiA

# Begin plotting the wavefunctions
clf()
contourf(x_values, z_values, psi, levels=20, cmap=cm.seismic)
axis('equal')