from ps5_problem2 import s, f, g, sAB, hAB, make_M, solve_eig
import numpy as np
from numpy import sqrt, pi, exp
import matplotlib.pyplot as plt
import pandas as pd


def g(uA, uB, alpha=2):
    """Returns the g(uA, uB) term."""
    return sqrt(0.5 * pi / alpha) * exp(-0.5 * alpha * (uA - uB) ** 2) * (3 / (16 * alpha ** 2) + (3 / (8 * alpha))
                                                                          * (uA + uB) ** 2 + (1 / 16) * (uA + uB) ** 4)


def gAB(uA, uB, vA, vB, alpha=2):
    """Returns the g_AB matrix element term."""
    return 2 * g(uA, uB, alpha) * g(vA, vB, alpha) - 2 * g(vA, uB, alpha) * g(uA, vB, alpha)


def make_HSG(n, center, alpha=2):
    """Returns the H, S and G matrix.

       Parameters
       ----------
       n  int, the size of H and S matrix
       center  array, an array of centers of basis functions
       alpha  int, width of basis functions.

       Returns
       -------
       H, S, G  2d array
       """
    S = np.zeros((n, n))
    H = np.zeros((n, n))
    G = np.zeros((n, n))
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            uA = center[i, 0]
            vA = center[i, 1]
            uB = center[j, 0]
            vB = center[j, 1]
            S[i, j] = sAB(uA, vA, uB, vB, alpha)
            H[i, j] = hAB(uA, vA, uB, vB, alpha) * 2
            G[i, j] = gAB(uA, uB, vA, vB, alpha)
    return H, S, G


def get_phi(x, y, ua, va, alpha):
    """Returns phi(x) = exp(-\alpha(x-xa)^2 - \alpha(y-va)^2) - exp(-\alpha(x-va)^2 - \alpha(y-ua)^2)
    Parameters
    ----------
    x  2d array
    y  2d array
    ua  int
    va  int
    Returns
    -------
    2d array
    """
    return np.exp(-alpha * (x - ua) ** 2 - alpha * (y - va) ** 2) - np.exp(
        -alpha * (x - va) ** 2 - alpha * (y - ua) ** 2)


def get_wfn(n, c, center, alpha):
    """Returns psi = \sum c_j phi_j
    Parameters
    ----------
    n  int, dimension of the grid where psi takes values over
    c  2d array, the coeff matrix
    M  2d array, the M matrix of center points
    i  int, the i-th energy level wfn

    Returns
    -------
    2d array
    """
    psi = 0
    #  grid points to query
    xq = np.linspace(-4, 4)
    yq = xq
    xx, yy = np.meshgrid(xq, yq)
    for j in range(n):
        uA = center[j, 0]
        vA = center[j, 1]
        psi = psi + c[j] * get_phi(xx, yy, uA, vA, alpha)
    return psi


if __name__ == '__main__':
    print("""i) See written part.""")

    # ii)
    # relevant parameters
    alpha = 2
    deltax = 0.5
    n = 10
    K = n * (2 * n + 1)
    center = make_M(n, deltax)
    H, S, G = make_HSG(K, center, alpha)
    avals = [0.001, 0.01, 0.1, 1, 10]
    Evals = []
    c0 = []
    for a in avals:
        Hnew = H + a * G
        E, c = solve_eig(Hnew, S)
        Evals.append(E[0])
        c0.append(c[:, 0])
    table = pd.DataFrame({'E0': Evals}, index=avals)
    table.index.name = 'a'
    print("""ii) The ground state energies for different a's are \n""", table)

    # iii)
    # gnd state
    avals.insert(0, 0)
    E, c = solve_eig(H, S)
    Evals.insert(0, E[0])
    c0.insert(0, c[:, 0])
    xq = np.linspace(-4, 4)
    yq = xq
    xx, yy = np.meshgrid(xq, yq)
    for i, a in enumerate(avals):
        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel='x', ylabel='y', title='Ground state wfn for a = {0}'.format(a))
        psi = get_wfn(K, c0[i], center, alpha)
        plt.contourf(xx, yy, psi, cmap='plasma')
        plt.axis('equal')
        plt.savefig('ps5_problem3iii{0}.png'.format(i))
        plt.show()

    # iv)
    avals = np.linspace(0, 0.01, 20)
    Evals = []
    for a in avals:
        Hnew = H + a * G
        E, c = solve_eig(Hnew, S)
        Evals.append(E[0])
    E_pertub = 2 + 45 / 16 * avals
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='Interaction strength', ylabel='$E_0$',
                         title='Ground state energy approximation comparison to pertubation theory')
    plt.plot(avals, Evals, label='Basis Set Expansion')
    plt.plot(avals, E_pertub, label='Pertubation theory')
    plt.legend()
    plt.savefig('ps5_problem3iv.png')
    plt.show()
