import numpy as np
from numpy.linalg import eig, inv
from numpy import sqrt
import matplotlib.pyplot as plt


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


def make_M(n, deltax=1):
    """Makes matrix M that has first column of [{-n}_{2n+1}, {-n+1}_{2n+1}]...
    and second column of [{-n, -n+1, ..., n}_{2n+1}] and first column is strictly
    greater than the second column element wise.

    Parameters
    ----------
    n int

    Returns
    -------
    2d array
    """
    query = np.arange(-n * deltax, n * deltax + deltax, deltax)
    xx, yy = np.meshgrid(query, query)
    c1 = xx.flatten('F')
    c2 = yy.flatten('F')
    M = np.column_stack((c1, c2))
    return M[M[:, 0] > M[:, 1]]


def s(uA, uB, alpha=2):
    """Returns the s(uA, uB) term."""
    return np.sqrt(0.5 * np.pi / alpha) * np.exp(-0.5 * alpha * (uA - uB) ** 2)


def f(uA, uB, alpha=2):
    """Returns the f(uA, uB) term."""
    return 0.5 * np.sqrt(0.5 * np.pi / alpha) * np.exp(-0.5 * alpha * (uA - uB) ** 2) \
           * (alpha - alpha ** 2 * (uA - uB) ** 2 + 0.25 * (1 / alpha + (uA + uB) ** 2))


def g(uA, uB, alpha=2):
    """Returns the g(uA, uB) term."""
    return np.sqrt(0.5 * np.pi / alpha) * np.exp(-0.5 * alpha * (uA - uB) ** 2) \
           * (3 / (16 * alpha ** 2) + (3 / (8 * alpha)) * (uA + uB) ** 2
              + (1 / 16) * (uA + uB) ** 4)


def sAB(uA, vA, uB, vB, alpha=2):
    """Returns the S_AB matrix element term."""
    return 2 * (s(uA, uB, alpha) * s(vA, vB, alpha) - s(vA, uB, alpha) * s(uA, vB, alpha))


def hAB(uA, vA, uB, vB, alpha=2):
    """Returns the H_AB matrix element term."""
    return f(uA, uB, alpha) * s(vA, vB, alpha) + f(vA, vB, alpha) \
           * s(uA, uB, alpha) - f(uA, vB, alpha) * s(vA, uB, alpha) - f(vA, uB, alpha) \
           * s(uA, vB, alpha)


def make_HS(n, center, alpha=2):
    """Returns the H and S matrix.

    Parameters
    ----------
    n  int, the size of H and S matrix
    center  array, an array of centers of basis functions
    alpha  int, width of basis functions.

    Returns
    -------
    H, S  2d array
    """
    S = np.zeros((n, n))
    H = np.zeros((n, n))
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            uA = center[i, 0]
            vA = center[i, 1]
            uB = center[j, 0]
            vB = center[j, 1]
            S[i, j] = sAB(uA, vA, uB, vB, alpha)
            H[i, j] = hAB(uA, vA, uB, vB, alpha) * 2
    return H, S


def solve_eig(H, S):
    """solves the generalized eigenvalue problem using scipy.linalg.eig. Sorts the evals and evecs from
    small to large and renormalizes each evec, such that the diagonal of c.T @ S @ c = 1.

    Parameters
    ----------
    H  2d array, the H matrix
    S  2d array, the S matrix

    Returns
    -------
    E  array, list of sorted e.vals.
    c  2d array, sorted evecs.
    """
    E, c = eig(inv(S) @ H)
    E = np.real(E)
    c = np.real(c)
    sorted = np.argsort(E)
    c_sorted = np.zeros((H.shape[0], H.shape[0]))
    for i in range(c_sorted.shape[0]):
        c_sorted[:, i] = c[:, sorted[i]]
        c_sorted[:, i] = c_sorted[:, i] / sqrt(c_sorted[:, i] @ S @ c_sorted[:, i])
    E = E[sorted]
    return E, c_sorted


if __name__ == '__main__':
    print("""See pdf for written part for i), iii), iv).""")

    # ii)
    xq = np.linspace(0, 3)
    yq = xq
    xx, yy = np.meshgrid(xq, yq)
    phi = get_phi(xx, yy, 2, 1, alpha=2)
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='x', ylabel='y', title='Contour plot of antisymmetric $phi$')
    plt.axis('equal')
    plt.contourf(xx, yy, phi, cmap='plasma')
    plt.savefig('ps5_problem2ii.png')
    plt.show()

    # v)
    # relevant parameters
    alpha = 2
    deltax = 0.5
    n = 10
    K = n * (2 * n + 1)
    center = make_M(n, deltax)
    H, S = make_HS(K, center, alpha)
    E, c = solve_eig(H, S)
    print("""v): The 9 lowest energies are: """, E[0:9])
