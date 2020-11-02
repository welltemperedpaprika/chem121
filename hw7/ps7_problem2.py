from numpy.linalg import eig, inv
from scipy.special import erf
import matplotlib.pyplot as plt
import numpy as np
from ps7_problem1 import compute_overlap, compute_kinetic_energy, \
    compute_elec_nuc_energy, compute_elec_elec_energy

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

def get_mullikenAnalysis(R):
    """
    Returns the mulliken population analysis for HeH+ given a internuclear distance.
    Parameters
    ----------
    R  float, internuclear distance

    Returns
    -------
    Mulliken analysis report, dict
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

    mulliken = {}
    N_total = np.diag(P @ S)
    mulliken['He'] = z_nuclei[0] - np.sum(N_total[0:2])
    mulliken['H'] = z_nuclei[1] - np.sum(N_total[-1])

    return mulliken

if __name__ == '__main__':
    #ii)
    print("""ii) The mulliken analysis for HeH+ at R=1.4 is {0}""".format(get_mullikenAnalysis(1.4)))

    #iii)
    r_array = np.linspace(1, 5, 20)  # internuclear distance array
    mulliken_He = [] # net charge on He
    mulliken_H = [] # net charge oh H
    for r in r_array:
        temp = get_mullikenAnalysis(r)
        mulliken_He.append(temp['He'])
        mulliken_H.append(temp['H'])
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='Internuclear distance ($a_0$)', ylabel='Mulliken net charge',
                         title='Mulliken net charge of HeH+ vs R')
    ax.plot(r_array, mulliken_H, label='H+')
    ax.plot(r_array, mulliken_He, label='He')
    plt.legend()
    plt.savefig('ps7_problem2iii.png')
    plt.show()
    print("""iii) The plot shows that at as the molecule gets separated, He becomes
    neutral, while H becomes a proton. i.e. Both electrons reside on He.""")


