import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.special import erf


def eigSinvH(S, H):
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
    SinvH = LA.inv(S) @ H
    E, U = LA.eig(SinvH)

    order = np.argsort(np.real(E))
    c = np.zeros((K, K))
    for i in range(K):
        c[:, i] = np.real(U[:, order[i]])
        c[:, i] = c[:, i] / np.sqrt(c[:, i] @ S @ c[:, i])
    E = np.sort(np.real(E))
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
        return 1 - x / 3
    else:
        return 0.5 * np.sqrt(np.pi) * erf(np.sqrt(x)) / np.sqrt(x)


def compute_overlap(alpha, beta, RA, RB):
    """
    Computes the overlap matrix element given by the formula in appendix A of Szabo & Ostlund.
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
    return (np.pi / absum) ** (3 / 2) * np.exp(-abfac * dRAB2)


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
    return (np.pi / absum) ** (3 / 2) * np.exp(-abfac * dRAB2) * \
           abfac * (3 - 2 * abfac * dRAB2)


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
    RP = (alpha * RA + beta * RB) / absum
    dRPC = RP - RC
    dRPC2 = dRPC @ dRPC
    return -(2 * np.pi / absum) * np.exp(-abfac * dRAB2) * \
           F0(absum * dRPC2)


if __name__ == '__main__':

    K = 3  # total basis function number
    L = 4  # number of Gaussians used to build each basis function

    # 2d array of width parameter. widths[i][j] gives the width of ith primitive gaussian of the
    # jth basis function
    widths = np.zeros((L, K))
    contraction_coeffs = np.zeros((L, K))  # we called these d
    centers = np.zeros((L, K, 3))  # initialize the 3d array corresponding to the xyz coord of each primitive gaussian.

    # Set width params
    widths[:, 0] = [0.42] * L
    widths[:, 1] = [0.045] * L
    widths[:, 2] = [0.014] * L

    # Set contraction coeffs
    contraction_coeffs[:, 0] = [1, -1] * 2
    contraction_coeffs[:, 1] = [1, -1] * 2
    contraction_coeffs[:, 2] = [1, -1] * 2

    offset = 0.01  # we called this Delta
    # Set the distance of each basis function away from the origin
    centers[0, :, :] = [offset, 0, offset]
    centers[1, :, :] = [offset, 0, -offset]
    centers[2, :, :] = [-offset, 0, -offset]
    centers[3, :, :] = [-offset, 0, offset]
    # Let the coord of the nucleus be 0,0,0
    R_nucleus = np.zeros(3)

    # Initialize overlap, kinetic, and nuclear matrix
    S = np.zeros((K, K))
    T = np.zeros((K, K))
    U = np.zeros((K, K))

    # Populate the matrix element using previous relevant functions into the corresponding matrices.
    for mu in range(K):
        for A in range(L):
            alpha = widths[A, mu]
            dAmu = contraction_coeffs[A, mu]
            RA = centers[A, mu, :]

            for nu in range(K):
                for B in range(L):
                    beta = widths[B, nu]
                    dBnu = contraction_coeffs[B, nu]
                    RB = centers[B, nu, :]
                    # we have to take contac. coeff. into account because our basis func. is made of linear comb.
                    # of primitive gaussians now.
                    S[mu, nu] += dAmu * dBnu * compute_overlap(alpha, beta, RA, RB)

                    T[mu, nu] += dAmu * dBnu * compute_kinetic_energy(alpha, beta, RA, RB)

                    U[mu, nu] += dAmu * dBnu * compute_elec_nuc_energy(alpha, beta, RA, RB, R_nucleus)

    H = T + U
    E, c = eigSinvH(S, H)  # Solves the eigenvalue equation

    #iv)
    print("""iv) The eigenvalues are {0}, and the eigenvectors are \n {1}""".format(E, c))

    #v)
    print("""v) The hydrogen energy levels are given by -1/(2n^2). For n=3, the exact
    energy is {0}, and our approximation is {1}. The relative error is {2}%, which means we fail
    to capture 1.6% of the exact energy.""".format(-1/(2*3**2), E[0],(E[0] + 1/(2*3**2))/(1/(2*3**2))*100))

    #vi)
    # Initialize relevant points to query for plotting purposes.
    x_values = np.arange(-10, 10, 0.1)
    z_values = x_values
    number_of_values = np.size(x_values)
    # Plot the ground state wavefunction
    state = 0
    psi = np.zeros((number_of_values, number_of_values))

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
                    phiA = np.exp(-alpha * ((x - xA) ** 2 + (z - zA) ** 2))

                    psi[i, j] += c[mu, state] * dAmu * phiA

    # Begin plotting the wavefunctions
    plt.clf()
    plt.contourf(x_values, z_values, psi, levels=20, cmap='plasma')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.axis('equal')
    plt.savefig('ps4_problem2vi.png')
    plt.show()