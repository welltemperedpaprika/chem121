import numpy as np
from numpy import array
from numpy.linalg import norm
from matplotlib import pyplot as plt
from ps7_problem1 import compute_overlap, compute_kinetic_energy, \
    compute_elec_nuc_energy, compute_elec_elec_energy, eigSinvH


def get_nrep_energy(R_nuclei, z_nuclei):
    """
    Returns the nuclear repulsion energy.
    Parameters
    ----------
    R_nuclei  array, Nuclei positions
    z_nuclei  array, Nuclei charges
    Returns
    -------
    E_nuclei  float, the nuclei nuclei repulsion energy
    """
    E_nuc = 0
    N_nuclei = R_nuclei.shape[0]
    for nucleus1 in range(N_nuclei):
        for nucleus2 in range(nucleus1 + 1, N_nuclei):
            dR = R_nuclei[nucleus1, :] - R_nuclei[nucleus2, :]
            dR2 = dR @ dR
            E_nuc += z_nuclei[nucleus1] * z_nuclei[nucleus2] / np.sqrt(dR2)

    return E_nuc


def make_basis(l_OH, a_HOH):
    """
    Returns a formatted nuclear positions, basis set info, for H2O given OH bond length, HOH angle.
    Parameters
    ----------
    l_OH  float, OH bond length
    a_HOH  float(radians), HOH bond angle

    Returns
    -------
    R_nuclei, widths, centers, contraction_coeffs
    """
    N_nuclei = 3  # number of atoms
    L = 2  # number of basis functions per contracted basis function
    K = 7  # number of contracted basis functions

    # widths of the basis functions
    zeta_H = 1.24
    zeta_O = 7.66

    # basis set coefficients
    one_s_coeff_1 = 0.164964
    one_s_coeff_2 = 0.381381
    two_s_coeff_1 = 0.168105
    two_s_coeff_2 = 0.0241442
    two_p_coeff_1 = 1.0
    two_p_coeff_2 = -1.0

    # part of the gaussian widths for each basis function
    one_s_width_1 = 0.151623
    one_s_width_2 = 0.851819
    two_s_width_1 = 0.493363
    two_s_width_2 = 1.945230
    two_p_width_1 = 0.9

    # offset in the exponent for the p orbital functions
    two_p_offset_1 = 0.1
    two_p_offset_2 = -0.1

    # place holders
    R_nuclei = np.zeros((N_nuclei, 3))
    widths = np.zeros((L, K))
    contraction_coeffs = np.zeros((L, K))
    centers = np.zeros((L, K, 3))

    # fill in our water coordinates. Oxygen is in the center, the hydrogen atoms are in the x-y plane below the oxygenatom
    R_nuclei[0, :] = [0, 0, 0]
    R_nuclei[1, :] = [l_OH * np.sin(a_HOH / 2), -l_OH * np.cos(a_HOH / 2), 0]
    R_nuclei[2, :] = [-l_OH * np.sin(a_HOH / 2), -l_OH * np.cos(a_HOH / 2), 0]

    # the orbitals are ordered 1s_H, 1s_H, 1s_O, 2s_O, 2p_O, 2p_O, 2p_O
    # widths of the contracted basis function components
    widths[:, 0] = [one_s_width_1 * zeta_H ** 2, one_s_width_2 * zeta_H ** 2]
    widths[:, 1] = [one_s_width_1 * zeta_H ** 2, one_s_width_2 * zeta_H ** 2]
    widths[:, 2] = [one_s_width_1 * zeta_O ** 2, one_s_width_2 * zeta_O ** 2]
    widths[:, 3] = [two_s_width_1, two_s_width_2]
    widths[:, 4] = [two_p_width_1, two_p_width_1]
    widths[:, 5] = [two_p_width_1, two_p_width_1]
    widths[:, 6] = [two_p_width_1, two_p_width_1]

    # coefficients for each contracted basis set component
    contraction_coeffs[:, 0] = [one_s_coeff_1, one_s_coeff_2]
    contraction_coeffs[:, 1] = [one_s_coeff_1, one_s_coeff_2]
    contraction_coeffs[:, 2] = [one_s_coeff_1, one_s_coeff_2]
    contraction_coeffs[:, 3] = [two_s_coeff_1, two_s_coeff_2]
    contraction_coeffs[:, 4] = [two_p_coeff_1, two_p_coeff_2]
    contraction_coeffs[:, 5] = [two_p_coeff_1, two_p_coeff_2]
    contraction_coeffs[:, 6] = [two_p_coeff_1, two_p_coeff_2]

    # for the centers keep in mind that the offset in the p-orbitals depends on which p orbital we are looking at
    # centers in the x direction
    centers[:, 0, 0] = [R_nuclei[1, 0], R_nuclei[1, 0]]
    centers[:, 1, 0] = [R_nuclei[2, 0], R_nuclei[2, 0]]
    centers[:, 2, 0] = [R_nuclei[0, 0], R_nuclei[0, 0]]
    centers[:, 3, 0] = [R_nuclei[0, 0], R_nuclei[0, 0]]
    centers[:, 4, 0] = [R_nuclei[0, 0] + two_p_offset_1, R_nuclei[0, 0] + two_p_offset_2]
    centers[:, 5, 0] = [R_nuclei[0, 0], R_nuclei[0, 0]]
    centers[:, 6, 0] = [R_nuclei[0, 0], R_nuclei[0, 0]]

    # centers in the y direction
    centers[:, 0, 1] = [R_nuclei[1, 1], R_nuclei[1, 1]]
    centers[:, 1, 1] = [R_nuclei[2, 1], R_nuclei[2, 1]]
    centers[:, 2, 1] = [R_nuclei[0, 1], R_nuclei[0, 1]]
    centers[:, 3, 1] = [R_nuclei[0, 1], R_nuclei[0, 1]]
    centers[:, 4, 1] = [R_nuclei[0, 1], R_nuclei[0, 1]]
    centers[:, 5, 1] = [R_nuclei[0, 1] + two_p_offset_1, R_nuclei[0, 1] + two_p_offset_2]
    centers[:, 6, 1] = [R_nuclei[0, 1], R_nuclei[0, 1]]

    # centers in the z direction
    centers[:, 0, 2] = [R_nuclei[1, 2], R_nuclei[1, 2]]
    centers[:, 1, 2] = [R_nuclei[2, 2], R_nuclei[2, 2]]
    centers[:, 2, 2] = [R_nuclei[0, 2], R_nuclei[0, 2]]
    centers[:, 3, 2] = [R_nuclei[0, 2], R_nuclei[0, 2]]
    centers[:, 4, 2] = [R_nuclei[0, 2], R_nuclei[0, 2]]
    centers[:, 5, 2] = [R_nuclei[0, 2], R_nuclei[0, 2]]
    centers[:, 6, 2] = [R_nuclei[0, 2] + two_p_offset_1, R_nuclei[0, 2] + two_p_offset_2]

    return R_nuclei, widths, centers, contraction_coeffs


def get_h2oSCFEnergy(l_OH, a_HOH):
    """
    Returns the total SCF energy for H2O for a given geometry.
    Parameters
    ----------
    l_OH  float, OH bond length
    a_HOH  float(radians), HOH bond angle

    Returns
    -------
    scf energy
    """
    N_nuclei = 3  # number of atoms
    L = 2  # number of basis functions per contracted basis function
    K = 7  # number of contracted basis functions
    N = 10
    R_nuclei, widths, centers, contraction_coeffs = make_basis(l_OH, a_HOH)
    z_nuclei = np.array([8, 1, 1])

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
        E_elec = sum(E[0:Nover2]) + 0.5 * np.trace(P @ h)
    E_nuc = get_nrep_energy(R_nuclei, z_nuclei)

    return E_elec + E_nuc


def get_h2oMulliken(l_OH, a_HOH):
    """
    Returns the Mulliken analysis for H2O for a given geometry.
    Parameters
    ----------
    l_OH  float, OH bond length
    a_HOH  float(radians), HOH bond angle

    Returns
    -------
    Mulliken analysis, dict
    """
    N_nuclei = 3  # number of atoms
    L = 2  # number of basis functions per contracted basis function
    K = 7  # number of contracted basis functions
    N = 10
    R_nuclei, widths, centers, contraction_coeffs = make_basis(l_OH, a_HOH)

    z_nuclei = np.array([8, 1, 1])

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
    mulliken = {}
    N_total = np.diag(P @ S)
    mulliken['O'] = z_nuclei[0] - np.sum(N_total[2::])
    mulliken['H1'] = z_nuclei[1] - np.sum(N_total[0])
    mulliken['H2'] = z_nuclei[1] - np.sum(N_total[1])

    return mulliken


def plot_contour(l_OH, a_HOH):
    """
    Plots the occupied orbital for H2O given a geometry
    Parameters
    ----------
    l_OH, bond length
    a_HOH, bond angle

    Returns
    -------
    None
    """
    N_nuclei = 3  # number of atoms
    L = 2  # number of basis functions per contracted basis function
    K = 7  # number of contracted basis functions
    N = 10
    R_nuclei, widths, centers, contraction_coeffs = make_basis(l_OH, a_HOH)

    z_nuclei = np.array([8, 1, 1])

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
    xq = np.linspace(-5, 5)
    yq = xq
    xx, yy = np.meshgrid(xq, yq)
    nocc = 5
    for i in range(nocc):
        psi = 0
        for j in range(K):
            gf1 = np.exp(-widths[:, j][0] * ((xx - centers[:, j, 0][0]) ** 2 + (yy - centers[:, j, 1][0]) ** 2))
            gf2 = np.exp(-widths[:, j][1] * ((xx - centers[:, j, 0][1]) ** 2 + (yy - centers[:, j, 1][1]) ** 2))
            phi = contraction_coeffs[:, j][0] * gf1 + contraction_coeffs[:, j][1] * gf2
            psi += c[j][i] * phi
        plt.clf()
        # add the atoms to our contour plot
        circles = []
        circles.append(plt.Circle((R_nuclei[0, 0], R_nuclei[0, 1]), 0.3, edgecolor='r', facecolor='r'))
        circles.append(plt.Circle((R_nuclei[1, 0], R_nuclei[1, 1]), 0.15, edgecolor='k', facecolor='w'))
        circles.append(plt.Circle((R_nuclei[2, 0], R_nuclei[2, 1]), 0.15, edgecolor='k', facecolor='w'))
        # the contour plot
        ax1 = plt.gca()
        for circle in circles:
            ax1.add_artist(circle)
        ax1.set_xlim((-5, 5))
        ax1.set_ylim((-5, 5))
        ax1.set_aspect('equal')
        plt.xlabel("x")
        plt.ylabel("y")
        plot_title = "{0}th occupied orbital".format(i)
        filename = "ps7_problem3v{0}thOrbital.png".format(i)
        plt.title(plot_title)
        plt.contourf(xx, yy, psi, levels=20, cmap=plt.cm.coolwarm)
        plt.savefig(filename)
        plt.show()


if __name__ == '__main__':
    # i)
    print("""i) Using the basis set from problem 6, the total SCF energy for water is {0}""".format(
        get_h2oSCFEnergy(1.809, np.deg2rad(104.52))))

    # ii)
    print("""ii) The Mulliken analysis on water is {0}. Since oxygen is more electronegative, 
    Mulliken analysis is not consistent with electronegativity.""".format(get_h2oMulliken(1.809, np.deg2rad(104.52))))

    # iii)
    r_array = np.linspace(1.5, 3, 20)  # find optimal OH length
    E_vals_r = []
    for r in r_array:
        E_vals_r.append(get_h2oSCFEnergy(r, np.deg2rad(104.52)))
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='OH distance ($a_0$)', ylabel='Total SCF energy (au)',
                         title='Total SCF energy of H2O vs R_OH')
    ax.plot(r_array, E_vals_r)
    plt.savefig('ps7_problem3iii.png')
    plt.show()
    print("""iii) The bond length that minimizes energy is {0}. It is longer than the correct result""".format(
        r_array[np.argmin(E_vals_r)]))

    # iv)
    angle_array = np.linspace(70, 120, 20) # find optimal HOH angle
    E_vals_a = []
    for a in angle_array:
        E_vals_a.append(get_h2oSCFEnergy(1.809, np.deg2rad(a)))
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, xlabel='HOH angle (degrees)', ylabel='Total SCF energy (au)',
                           title='Total SCF energy of H2O vs HOH angle')
    ax1.plot(angle_array, E_vals_a)
    plt.savefig('ps7_problem3iv.png')
    plt.show()
    print("""iii) The bond angle that minimizes energy is {0}. It is smaller than the actual result""".format(
        angle_array[np.argmin(E_vals_a)]))

    # v)
    min_r = r_array[np.argmin(E_vals_r)]
    min_angle = angle_array[np.argmin(E_vals_a)]
    plot_contour(min_r, np.deg2rad(min_angle))
