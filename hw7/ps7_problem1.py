import numpy as np
from numpy.linalg import eig, inv
from scipy.special import erf
import matplotlib.pyplot as plt

def eigSinvH(S, H, K):
    SinvH = inv(S) @ H
    E, U = eig(SinvH)

    order = np.argsort(np.real(E))
    c = np.zeros((K, K))
    for i in range(K):
        c[:, i] = np.real(U[:, order[i]])
        c[:, i] = c[:, i] / np.sqrt(c[:, i] @ S @ c[:, i])
    E = np.sort(np.real(E))
    return E, c

def F0(x):
    if (x < 1e-8):
        return 1 - x/3
    else:
        return 0.5*np.sqrt(np.pi)*erf(np.sqrt(x))/np.sqrt(x)

def compute_overlap(alpha, beta, RA, RB):
    absum = alpha + beta
    abfac = alpha * beta / absum
    dRAB = RA - RB
    dRAB2 = dRAB @ dRAB
    return (np.pi/absum)**(3/2) * np.exp(-abfac * dRAB2)

def compute_kinetic_energy(alpha, beta, RA, RB):
    absum = alpha + beta
    abfac = alpha * beta / absum
    dRAB = RA - RB
    dRAB2 = dRAB @ dRAB
    return (np.pi/absum)**(3/2) * np.exp(-abfac * dRAB2) * \
            abfac * (3 - 2*abfac * dRAB2)

def compute_elec_nuc_energy(alpha, beta, RA, RB, RC):
    absum = alpha + beta
    abfac = alpha * beta / absum
    dRAB = RA - RB
    dRAB2 = dRAB @ dRAB
    RP = (alpha*RA + beta*RB)/absum
    dRPC = RP - RC
    dRPC2 = dRPC @ dRPC
    return -(2*np.pi/absum)*np.exp(-abfac * dRAB2) * \
            F0(absum * dRPC2)

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

    return 2 * np.pi**(5/2) * (absum * gdsum * np.sqrt(abgdsum))**(-1) * \
            np.exp(-abfac*dRAB2 - gdfac*dRCD2) * \
            F0(abgdfac * dRPQ2)

def get_hehScfEnergy(R):
    """
    Returns the total SCF energy for HeH+ for a given internuclear distance.
    Parameters
    ----------
    R  float, internuclear distance

    Returns
    -------
    scf energy
    """
    N = 2
    N_nuclei = 2
    z_nuclei = np.array([2, 1])
    K = 2
    L = 3  # number of Gaussians used to build each basis function

    widths = np.zeros((L, K))
    contraction_coeffs = np.zeros((L, K))  # we called these d
    centers = np.zeros((L, K, 3))

    alpha1s_STO3G = np.array([0.109818, 0.405771, 2.22766])
    d1s_STO3G = np.array([0.316894, 0.381531, 0.109991])

    # 1s on He
    zeta_He = 2.0925
    widths[:, 0] = alpha1s_STO3G * zeta_He ** 2
    contraction_coeffs[:, 0] = d1s_STO3G * widths[:, 0] ** (3 / 4)

    # 1s on H
    zeta_H = 1.24
    widths[:, 1] = alpha1s_STO3G * zeta_H ** 2
    contraction_coeffs[:, 1] = d1s_STO3G * widths[:, 1] ** (3 / 4)
    R_nuclei = np.zeros((N_nuclei,3))
    # R = 1.4632
    R_nuclei[1,0] = R

    centers[0,1,0] = R
    centers[1,1,0] = R
    centers[2,1,0] = R

    S = np.zeros((K,K))
    T = np.zeros((K,K))
    U1 = np.zeros((K,K))
    U2 = np.zeros((K,K,K,K))

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

    h = T + U1
    E, c = eigSinvH(S,h,K)
    Nover2 = int(N/2)

    n_iterations = 10
    for iterate in range(n_iterations):
        P = np.zeros((K,K))
        for mu in range(K):
            for nu in range(K):
                for j in range(Nover2):
                    P[mu,nu] += 2 * c[mu,j] * c[nu,j]

        F = np.copy(h)
        for mu in range(K):
            for nu in range(K):
                for lam in range(K):
                    for sigma in range(K):
                        F[mu,nu] += P[lam,sigma] * \
                                    ( U2[mu,nu,lam,sigma] - 0.5*U2[mu,sigma,lam,nu] )

        E, c = eigSinvH(S,F,K)

        E_elec = sum(E[0:Nover2]) + 0.5 * np.trace(P @ h)

    # energy of nuclear repulsion
    E_nuc = 0
    for nucleus1 in range(N_nuclei):
        for nucleus2 in range(nucleus1+1,N_nuclei):
            dR = R_nuclei[nucleus1,:] - R_nuclei[nucleus2,:]
            dR2 = dR @ dR
            E_nuc += z_nuclei[nucleus1] * z_nuclei[nucleus2] / np.sqrt(dR2)

    E_total = E_elec + E_nuc
    return E_total

def get_heKoopmans(i):
    """
    Returns the ith ionization energy given by Koopmans theorem.
    Parameters
    ----------
    i  int

    Returns
    -------
    ionization energy
    """
    N = 2
    N_nuclei = 2
    z_nuclei = np.array([2, 1])
    K = 2
    L = 3  # number of Gaussians used to build each basis function
    R = 10000
    widths = np.zeros((L, K))
    contraction_coeffs = np.zeros((L, K))  # we called these d
    centers = np.zeros((L, K, 3))

    alpha1s_STO3G = np.array([0.109818, 0.405771, 2.22766])
    d1s_STO3G = np.array([0.316894, 0.381531, 0.109991])

    # 1s on He
    zeta_He = 2.0925
    widths[:, 0] = alpha1s_STO3G * zeta_He ** 2
    contraction_coeffs[:, 0] = d1s_STO3G * widths[:, 0] ** (3 / 4)

    # 1s on H
    zeta_H = 1.24
    widths[:, 1] = alpha1s_STO3G * zeta_H ** 2
    contraction_coeffs[:, 1] = d1s_STO3G * widths[:, 1] ** (3 / 4)
    R_nuclei = np.zeros((N_nuclei, 3))
    # R = 1.4632
    R_nuclei[1, 0] = R

    centers[0, 1, 0] = R
    centers[1, 1, 0] = R
    centers[2, 1, 0] = R

    S = np.zeros((K, K))
    T = np.zeros((K, K))
    U1 = np.zeros((K, K))
    U2 = np.zeros((K, K, K, K))

    # compute matrix elements
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

                    S[mu, nu] += dAmu * dBnu * compute_overlap(alpha, beta, RA, RB)

                    T[mu, nu] += dAmu * dBnu * compute_kinetic_energy(alpha, beta, RA, RB)

                    for nucleus in range(N_nuclei):
                        U1[mu, nu] += dAmu * dBnu * z_nuclei[nucleus] * \
                                      compute_elec_nuc_energy(alpha, beta, RA, RB, R_nuclei[nucleus])

                    for sigma in range(K):
                        for C in range(L):
                            gamma = widths[C, sigma]
                            dCsigma = contraction_coeffs[C, sigma]
                            RC = centers[C, sigma, :]

                            for lam in range(K):
                                for D in range(L):
                                    delta = widths[D, lam]
                                    dDlam = contraction_coeffs[D, lam]
                                    RD = centers[D, lam, :]

                                    U2[mu, nu, sigma, lam] += dAmu * dBnu * dCsigma * dDlam * \
                                                              compute_elec_elec_energy(alpha, beta, gamma, delta, RA,
                                                                                       RB, RC, RD)

    h = T + U1
    E, c = eigSinvH(S, h, K)

    Nover2 = int(N / 2)

    n_iterations = 10
    for iterate in range(n_iterations):
        P = np.zeros((K, K))
        for mu in range(K):
            for nu in range(K):
                for j in range(Nover2):
                    P[mu, nu] += 2 * c[mu, j] * c[nu, j]

        F = np.copy(h)
        for mu in range(K):
            for nu in range(K):
                for lam in range(K):
                    for sigma in range(K):
                        F[mu, nu] += P[lam, sigma] * \
                                     (U2[mu, nu, lam, sigma] - 0.5 * U2[mu, sigma, lam, nu])

        E, c = eigSinvH(S, F, K)
    return -E[i]

def get_improvedHehScfenergy(R):
    """
    Returns a improved SCF energy by adding a new basis for HeH+ for a given internuclear distance
    Parameters
    ----------
    R  float, internuclear distance

    Returns
    -------
    total scf enegry
    """
    N = 2
    N_nuclei = 2
    z_nuclei = np.array([2, 1])
    K = 3
    L = 3  # number of Gaussians used to build each basis function

    widths = np.zeros((L, K))
    contraction_coeffs = np.zeros((L, K))  # we called these d
    centers = np.zeros((L, K, 3))

    alpha1s_STO3G = np.array([0.109818, 0.405771, 2.22766])
    d1s_STO3G = np.array([0.316894, 0.381531, 0.109991])

    # 1s on He
    zeta_He = 2.0925
    widths[:, 0] = alpha1s_STO3G * zeta_He ** 2
    contraction_coeffs[:, 0] = d1s_STO3G * widths[:, 0] ** (3 / 4)

    zeta_He_new = 1.6875
    widths[:, 1] = alpha1s_STO3G * zeta_He_new ** 2
    contraction_coeffs[:, 1] = d1s_STO3G * widths[:, 1] ** (3 / 4)
    # 1s on H
    zeta_H = 1.24
    widths[:, 2] = alpha1s_STO3G * zeta_H ** 2
    contraction_coeffs[:, 2] = d1s_STO3G * widths[:, 2] ** (3 / 4)


    # new basis for He


    # R = 1.4632
    R_nuclei = np.zeros((N_nuclei, 3))
    R_nuclei[1, 0] = R

    centers[0, 2, 0] = R
    centers[1, 2, 0] = R
    centers[2, 2, 0] = R

    S = np.zeros((K, K))
    T = np.zeros((K, K))
    U1 = np.zeros((K, K))
    U2 = np.zeros((K, K, K, K))

    # compute matrix elements
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

                    S[mu, nu] += dAmu * dBnu * compute_overlap(alpha, beta, RA, RB)

                    T[mu, nu] += dAmu * dBnu * compute_kinetic_energy(alpha, beta, RA, RB)

                    for nucleus in range(N_nuclei):
                        U1[mu, nu] += dAmu * dBnu * z_nuclei[nucleus] * \
                                      compute_elec_nuc_energy(alpha, beta, RA, RB, R_nuclei[nucleus])

                    for sigma in range(K):
                        for C in range(L):
                            gamma = widths[C, sigma]
                            dCsigma = contraction_coeffs[C, sigma]
                            RC = centers[C, sigma, :]

                            for lam in range(K):
                                for D in range(L):
                                    delta = widths[D, lam]
                                    dDlam = contraction_coeffs[D, lam]
                                    RD = centers[D, lam, :]

                                    U2[mu, nu, sigma, lam] += dAmu * dBnu * dCsigma * dDlam * \
                                                              compute_elec_elec_energy(alpha, beta, gamma, delta, RA,
                                                                                       RB, RC, RD)

    h = T + U1
    E, c = eigSinvH(S,h,K)

    Nover2 = int(N / 2)

    n_iterations = 10
    for iterate in range(n_iterations):
        P = np.zeros((K, K))
        for mu in range(K):
            for nu in range(K):
                for j in range(Nover2):
                    P[mu, nu] += 2 * c[mu, j] * c[nu, j]

        F = np.copy(h)
        for mu in range(K):
            for nu in range(K):
                for lam in range(K):
                    for sigma in range(K):
                        F[mu, nu] += P[lam, sigma] * \
                                     (U2[mu, nu, lam, sigma] - 0.5 * U2[mu, sigma, lam, nu])

        E, c = eigSinvH(S,F,K)

        E_elec = sum(E[0:Nover2]) + 0.5 * np.trace(P @ h)

    # energy of nuclear repulsion
    E_nuc = 0
    for nucleus1 in range(N_nuclei):
        for nucleus2 in range(nucleus1 + 1, N_nuclei):
            dR = R_nuclei[nucleus1, :] - R_nuclei[nucleus2, :]
            dR2 = dR @ dR
            E_nuc += z_nuclei[nucleus1] * z_nuclei[nucleus2] / np.sqrt(dR2)

    E_total = E_elec + E_nuc
    return E_total

def get_improvedHeKoopmans(i):
    """
    Returns a improved ith ionization energy value by adding a new basis.
    Parameters
    ----------
    i

    Returns
    -------
    ionization energy
    """
    N = 2
    N_nuclei = 2
    z_nuclei = np.array([2, 1])
    K = 3
    L = 3  # number of Gaussians used to build each basis function
    R = 1000
    widths = np.zeros((L, K))
    contraction_coeffs = np.zeros((L, K))  # we called these d
    centers = np.zeros((L, K, 3))

    alpha1s_STO3G = np.array([0.109818, 0.405771, 2.22766])
    d1s_STO3G = np.array([0.316894, 0.381531, 0.109991])

    # 1s on He
    zeta_He = 2.0925
    widths[:, 0] = alpha1s_STO3G * zeta_He ** 2
    contraction_coeffs[:, 0] = d1s_STO3G * widths[:, 0] ** (3 / 4)

    zeta_He_new = 1.6875
    widths[:, 1] = alpha1s_STO3G * zeta_He_new ** 2
    contraction_coeffs[:, 1] = d1s_STO3G * widths[:, 1] ** (3 / 4)
    # 1s on H
    zeta_H = 1.24
    widths[:, 2] = alpha1s_STO3G * zeta_H ** 2
    contraction_coeffs[:, 2] = d1s_STO3G * widths[:, 2] ** (3 / 4)


    # new basis for He


    # R = 1.4632
    R_nuclei = np.zeros((N_nuclei, 3))
    R_nuclei[1, 0] = R

    centers[0, 2, 0] = R
    centers[1, 2, 0] = R
    centers[2, 2, 0] = R

    S = np.zeros((K, K))
    T = np.zeros((K, K))
    U1 = np.zeros((K, K))
    U2 = np.zeros((K, K, K, K))

    # compute matrix elements
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

                    S[mu, nu] += dAmu * dBnu * compute_overlap(alpha, beta, RA, RB)

                    T[mu, nu] += dAmu * dBnu * compute_kinetic_energy(alpha, beta, RA, RB)

                    for nucleus in range(N_nuclei):
                        U1[mu, nu] += dAmu * dBnu * z_nuclei[nucleus] * \
                                      compute_elec_nuc_energy(alpha, beta, RA, RB, R_nuclei[nucleus])

                    for sigma in range(K):
                        for C in range(L):
                            gamma = widths[C, sigma]
                            dCsigma = contraction_coeffs[C, sigma]
                            RC = centers[C, sigma, :]

                            for lam in range(K):
                                for D in range(L):
                                    delta = widths[D, lam]
                                    dDlam = contraction_coeffs[D, lam]
                                    RD = centers[D, lam, :]

                                    U2[mu, nu, sigma, lam] += dAmu * dBnu * dCsigma * dDlam * \
                                                              compute_elec_elec_energy(alpha, beta, gamma, delta, RA,
                                                                                       RB, RC, RD)

    h = T + U1
    E, c = eigSinvH(S,h,K)

    Nover2 = int(N / 2)

    n_iterations = 10
    for iterate in range(n_iterations):
        P = np.zeros((K, K))
        for mu in range(K):
            for nu in range(K):
                for j in range(Nover2):
                    P[mu, nu] += 2 * c[mu, j] * c[nu, j]

        F = np.copy(h)
        for mu in range(K):
            for nu in range(K):
                for lam in range(K):
                    for sigma in range(K):
                        F[mu, nu] += P[lam, sigma] * \
                                     (U2[mu, nu, lam, sigma] - 0.5 * U2[mu, sigma, lam, nu])

        E, c = eigSinvH(S,F,K)

        return -E[i]

if __name__ == '__main__':
    r_array = np.linspace(0.5, 10, 100) # set internuclear distance array.
    E_vals = []
    for r in r_array:
        E_vals.append(get_hehScfEnergy(r))
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='Internuclear distance ($a_0$)', ylabel='Total SCF energy (au)',
                         title='Total SCF energy of HeH+ vs R')
    ax.plot(r_array, E_vals)
    plt.savefig('ps7_problem1ii.png')
    plt.show()
    E_diss = E_vals[-1] - np.min(E_vals)
    print("""ii) The dissociation energy is {0} hartree""".format(E_diss))

    #iii)
    print("""iii) Our dissociation energy is {0} ha, which is {1}kcal/mol. Compared to the exact
    value of 47 kcal/mol, our number is off by almost 300%.""".format(E_diss, E_diss * 630))

    #iv)
    print("""iv) The ionization energy for a helium atom according to Koopman's theorem is {0} kcal/mol""".format(
        get_heKoopmans(0) * 630)
    )

    #v)
    print("""v) The ionization energy is smaller than the actual ionization energy.""")

    #vi)
    print("""vi) By adding a new basis function, the new equilibrium energy is {0}""".format(get_improvedHehScfenergy(
        1.4632)))

    #vii)
    E_diss_new = get_improvedHehScfenergy(1000) - get_improvedHehScfenergy(1.4632)
    print("""vii) The new dissociation energy is {0} kcal/mol, which is a much better 
    improvement than the previous one.""".format(E_diss_new * 630))

    #viii)
    E_ionization_new = get_improvedHeKoopmans(0)
    print("""viii) The new ionization energy is {0} kcal/mol, according to Koopmans theorem. The 
    estimate has also significantly improved, but still about 70 kcal away from the true value.""".format(
        E_ionization_new * 630))

