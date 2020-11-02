import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, eig
from scipy.special import comb
from ps5_problem3 import make_HSG
from ps5_problem2 import make_M

def eigSinvH(S,H,K):
    SinvH = inv(S) @ H
    E, U = eig(SinvH)

    order = np.argsort(E)
    c = np.zeros((K, K))
    for i in range(K):
        c[:, i] = U[:, order[i]]
        c[:, i] = c[:, i] / np.sqrt(c[:, i] @ S @ c[:, i])

    E = np.sort(E)

    return E, c

def do_scf(a):
    """does the scf procedure given interaction strength a."""
    alpha = 2
    deltax = 0.5

    n = 10
    K = 2 * n + 1

    center = np.arange(-n * deltax, (n + 1) * deltax, deltax)

    S = np.zeros((K, K))
    h = np.zeros((K, K))
    G = np.zeros((K, K))

    for A in range(K):
        xA = center[A]
        for B in range(K):
            xB = center[B]
            #  populate overlap matrix element
            S[A, B] = np.sqrt(0.5 * np.pi / alpha) * np.exp(-0.5 * alpha * (xA - xB) ** 2)
            #  populate hamiltonian matrix element
            h[A, B] = 0.5 * S[A, B] * (alpha - alpha ** 2 * (xA - xB) ** 2 + \
                                       0.25 * (1 / alpha + (xA + xB) ** 2))
            #  populate g matrix element
            G[A, B] = S[A, B] * (3 / (16 * alpha ** 2) + \
                                 (3 / (8 * alpha)) * (xA + xB) ** 2 + \
                                 (1 / 16) * (xA + xB) ** 4)

    # initial guess for the eigenvalue and eigenvector
    E, c = eigSinvH(S, h, K)
    c1 = c[:, 0]
    c2 = c[:, 1]

    # begin scf procedure
    P = np.zeros((K, K))
    #  set limit of SCF iterations, assume to converge
    niterations = 50
    for iteration in range(niterations):
        for D in range(K):
            for E in range(K):
                P[D, E] = c2[D] * c2[E] + c1[D] * c1[E]
        # P = outer(c1,c1) + outer(c2,c2)
        #  construct the "fock" matrix
        heff = h + a * (np.trace(P @ G) * G - G @ P @ G)
        # update the eigenvalue and eigenvectors
        E, c = eigSinvH(S, heff, K)
        c1 = c[:, 0]
        c2 = c[:, 1]
        e1 = E[0]
        e2 = E[1]
        # prints the total energy of the system
        Etot = 0.5 * (e1 + e2 + np.trace(P @ h))
    return Etot, S, h, G, c

def do_CI(N_orb, a):
    """does the CI procedure given N_orb, number of orbitals, and a, interaction strength."""
    N_config = int(comb(N_orb, 2))  # number of configurations possible, ignoring spin
    Etot, _, h, G, c = do_scf(a)
    config_list = np.zeros((N_config, 2)).astype(int)  # initialize possible configs as [i,j]th occupied orbitals
    count = 0
    for i in range(N_orb):
        for j in range(i + 1, N_orb):
            config_list[count, :] = [i, j]
            count += 1

    # initialize CI matrix
    H_CI = np.zeros((N_config, N_config))

    # populate CI matrix elements \bra \psi_ij | H | \psi_ij \ket, where \psi is a configuration
    for config1 in range(N_config):
        i, j = config_list[config1, :]
        for config2 in range(N_config):
            k, l = config_list[config2, :]
            if i == l:
                H_CI[config1, config2] -= c[:, j] @ h @ c[:, k]
            if j == k:
                H_CI[config1, config2] -= c[:, i] @ h @ c[:, l]
            if j == l:
                H_CI[config1, config2] += c[:, i] @ h @ c[:, k]
            if i == k:
                H_CI[config1, config2] += c[:, j] @ h @ c[:, l]

            H_CI[config1, config2] += a * ((c[:, i] @ G @ c[:, k]) * (c[:, j] @ G @ c[:, l])
                                           - (c[:, j] @ G @ c[:, k]) * (c[:, i] @ G @ c[:, l]))

    E, b = eig(H_CI)
    order = np.argsort(E)
    E = np.real(np.sort(E))
    return E, b, order, config_list

def get_bse_energy(a):
    """Returns the basis set expansion energy given interaction strength a."""
    alpha = 2
    deltax = 0.5
    n = 10
    K = n * (2 * n + 1)
    center = make_M(n, deltax)
    H, S, G = make_HSG(K, center, alpha)
    Hnew = H + a * G
    E, _ = eigSinvH(S, Hnew, K)
    return E

if __name__ == '__main__':
    #i, ii, iii)
    print("""See pdf for written solution for part i and ii.""")

    #iv)
    N_orb = 16
    a_vals = np.arange(0, 101, 3)
    E_vals_CI = []
    E_vals_exact = []
    '''for a in a_vals:
        E_CI, _, _, _ = do_CI(N_orb, a)
        E_vals_CI.append(E_CI[0])
        E_bse = get_bse_energy(a)
        E_vals_exact.append(E_bse[0])
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='interaction strength', ylabel='energy (a.u)',
                         title='Configuration Interaction energy vs basis set expansion energy')
    ax.plot(a_vals, E_vals_CI, label='CI energy')
    ax.plot(a_vals, E_vals_exact,label='BSE solution')
    plt.legend()
    plt.savefig('ps8_problem3iv.png')
    plt.show()'''

    #v)
    print("""v) Here we use 0 indexing. I would expect (0, 1), (1, 2), and (2, 3) to contribute the most due to the 
    symmetry restriction and the pertubation theory formula.""")

    #vi)
    _, b, order, config_list = do_CI(16, 0.01)
    coeffs = b[:, order[0]]
    ind = np.argpartition(abs(coeffs), -5)[-5:]
    ind = ind[np.argsort(abs(coeffs[ind]))][::-1]
    top5coeffs = coeffs[ind]
    top5configs = [config_list[i, :] for i in ind]
    print("""vi) Here we use zero indexing. The top five coefficients are {0}, with corresponding configurations {1}. 
    Apart from the ground state config itself, we see that among the lowest five energy single orbitals, (2, 
    3) and (3, 4) contribute the most. This is a little different than expected.""".
          format(top5coeffs, top5configs))

    #vii)
    _, b, order, config_list = do_CI(16, 1)
    coeffs = b[:, order[0]]
    ind = np.argpartition(abs(coeffs), -5)[-5:]
    ind = ind[np.argsort(abs(coeffs[ind]))][::-1]
    top5coeffs = coeffs[ind]
    top5configs = [config_list[i, :] for i in ind]
    print("""vii) Here we use zero indexing. The top five coefficients are {0}, with corresponding configurations {1}. 
        Apart from the ground state config itself, we see that the among the lowest five energy single orbitals,
         lower energy configurations that obey the symmetry requirement, e.g. (2, 3) and (1, 2), contribute the most, 
         which is consistent with what was expected.""".
          format(top5coeffs, top5configs))

