import numpy as np
from numpy.linalg import inv, eig


def eigSinvH(S, H, K):
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

    #  calculate matrix elements
    for A in range(K):
        xA = center[A]
        for B in range(K):
            xB = center[B]

            S[A, B] = np.sqrt(0.5 * np.pi / alpha) * np.exp(-0.5 * alpha * (xA - xB) ** 2)

            h[A, B] = 0.5 * S[A, B] * (alpha - alpha ** 2 * (xA - xB) ** 2 + \
                                       0.25 * (1 / alpha + (xA + xB) ** 2))

            G[A, B] = S[A, B] * (3 / (16 * alpha ** 2) + \
                                 (3 / (8 * alpha)) * (xA + xB) ** 2 + \
                                 (1 / 16) * (xA + xB) ** 4)

    E, c = eigSinvH(S, h, K)
    c1 = c[:, 0]
    c2 = c[:, 0]

    # begin scf procedure
    niterations = 100
    for iteration in range(niterations):
        avx4 = c1 @ G @ c1
        avy4 = c2 @ G @ c2

        heffx = h + a * avy4 * G
        heffy = h + a * avx4 * G

        E, c = eigSinvH(S, heffx, K)
        c1 = c[:, 0]
        e1 = E[0]

        E, c = eigSinvH(S, heffy, K)
        c2 = c[:, 0]
        e2 = E[0]

        Etot = e1 + e2 - a * avx4 * avy4
    return Etot, S, h, G, c


def do_CI(N_orb, a):
    """does the CI procedure given N_orb, number of orbitals, and a, interaction strength."""
    N_config = N_orb ** 2  # number of configurations possible, ignoring spin
    _, _, h, G, c = do_scf(a)
    config_list = np.zeros((N_config, 2)).astype(int)  # initialize possible configs as [i,j]th occupied orbitals
    count = 0
    for i in range(N_orb):
        for j in range(N_orb):
            config_list[count, :] = [i, j]
            count += 1

    # initialize CI matrix
    H_CI = np.zeros((N_config, N_config))

    # populate CI matrix elements \bra \psi_ij | H | \psi_ij \ket, where \psi is a configuration
    for config1 in range(N_config):
        i, j = config_list[config1, :]
        for config2 in range(N_config):
            k, l = config_list[config2, :]

            if j == l:
                H_CI[config1, config2] += c[:, i] @ h @ c[:, k]
            if i == k:
                H_CI[config1, config2] += c[:, j] @ h @ c[:, l]

            H_CI[config1, config2] += a * (c[:, k] @ G @ c[:, i]) * (c[:, l] @ G @ c[:, j])

    E, b = eig(H_CI)
    order = np.argsort(E)
    E = np.real(np.sort(E))
    return E, b, order, config_list


if __name__ == '__main__':
    # i)
    print("""See interacting_oscillators_CI.py for comments.""")

    # ii)
    _, b, order, config_list = do_CI(5, 0.01)
    print("""The coefficients are, in order of their energy, {0}""".format(b[:, order[0]]))

    # iii)
    coeffs = b[:, order[0]]
    ind = np.argpartition(abs(coeffs), -4)[-4:]
    ind = ind[np.argsort(abs(coeffs[ind]))][::-1]
    top4coeffs = coeffs[ind]
    top4configs = [config_list[i, :] for i in ind]
    print("""Here we use zero indexing. The top four coefficients are {0}, with corresponding configurations {1}. Apart 
    from the ground
    state config itself, we see that even configurations contribute the most, which is consistent with what was 
    expected.""".
          format(top4coeffs, top4configs))

    # iv)
    _, b, order, config_list = do_CI(5, 1)
    print("""The coefficients are, in order of their energy, {0}""".format(b[:, order[0]]))
    coeffs = b[:, order[0]]
    ind = np.argpartition(abs(coeffs), -4)[-4:]
    ind = ind[np.argsort(abs(coeffs[ind]))][::-1]
    top4coeffs = coeffs[ind]
    top4configs = [config_list[i, :] for i in ind]
    print("""Here we use zero indexing. The top four coefficients are {0}, with corresponding configurations {1}. 
    This time we see that
    (0, 2), (2, 0), (2, 2) to be most contributing. This is not really consistent with what I expected in 
    question 1 because there are many contributions from singly excited configurations, which we saw would be 0 in 
    the energy contribution. Hence, pertubation theory should be viewed with caution at large interaction limits.""".
          format(top4coeffs, top4configs))
