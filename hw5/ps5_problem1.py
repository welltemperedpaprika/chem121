import numpy as np
from numpy.linalg import eig, inv
from numpy import sqrt, pi, exp
import matplotlib.pyplot as plt
import pandas as pd


def make_M(n, deltax=1):
    """Makes matrix M that has first column of [{-n}_{2n+1}, {-n+1}_{2n+1}]...
    and second column of [{-n, -n+1, ..., n}_{2n+1}]

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
    return M


def make_S(n, x, alpha):
    """makes the overlap matrix. Implementation utilizes numpy's element wise operation. rows and cols contain
    the indices of S_{AB} in the form of row: [{0}_n, {1}_n, ...], col: [{0, 1..,n}_n] such that they extract
    x and y in the form of x[rows]:[{x0}_n, {x1}_n...] etc. The result is then reshaped to an nxn matrix.

    Parameters
    ----------
    n  int, dimension of resulting S matrix
    alpha  int, the width factor
    x  1d array, an array of x indices

    Returns
    -------
    2d array
    """
    indices = np.arange(n)
    rows, cols = np.meshgrid(indices, indices)
    rows = rows.flatten('F')
    cols = cols.flatten('F')
    S = sqrt(pi / (2 * alpha)) * exp(-(alpha / 2) * (x[rows] - x[cols]) ** 2)
    S = np.reshape(S, (n, n))
    return S


def make_G(n, x, S, alpha):
    """makes the G matrix. Implementation utilizes numpy's element wise operation. rows and cols contain
    the indices of H_{AB} in the form of row: [{0}_n, {1}_n, ...], col: [{0, 1..,n}_n] such that they extract
    x and y in the form of x[rows]:[{x0}_n, {x1}_n...] etc. The result is then reshaped to an nxn matrix.

    Parameters
    ----------
    n  int, dimension of resulting H matrix
    alpha  int, the width factor
    x  1d array, an array of x indices
    S  2d array, the S matrix
    Returns
    -------
    2d array
    """
    indices = np.arange(n)
    rows, cols = np.meshgrid(indices, indices)
    rows = rows.flatten('F')
    cols = cols.flatten('F')
    S = S.flatten()
    G = S * (3 / (16 * alpha ** 2) + 3 / (8 * alpha) * (x[rows] + x[cols]) ** 2 + 1 / 16 * (x[rows] + x[cols]) ** 4)
    G = np.reshape(G, (n, n))
    return G


def make_H(n, x, alpha):
    """makes the hamiltonian matrix. Implementation utilizes numpy's element wise operation. rows and cols contain
    the indices of H_{AB} in the form of row: [{0}_n, {1}_n, ...], col: [{0, 1..,n}_n] such that they extract
    x and y in the form of x[rows]:[{x0}_n, {x1}_n...] etc. The result is then reshaped to an nxn matrix.
    Parameters
    ----------
    n  int, dimension of resulting H matrix
    a  int, the interaction factor
    alpha  int, the width factor
    x  1d array, an array of x indices

    Returns
    -------
    2d array
    """
    S = make_S(n, x, alpha)
    indices = np.arange(n)
    rows, cols = np.meshgrid(indices, indices)
    rows = rows.flatten('F')
    cols = cols.flatten('F')
    S = S.flatten()
    H = 0.5 * S * (alpha - alpha ** 2 * (x[rows] - x[cols]) ** 2
                   + 0.25 * (1 / alpha + (x[rows] + x[cols]) ** 2))
    H = np.reshape(H, (n, n))
    return H


def solve_eig(H, S):
    """solves the generalized eigenvalue problem using numpy.linalg.eig. Sorts the evals and evecs from
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


def do_scf(h, G, S, a, niterations=100):
    """Computes the SCF solution given H, G, S, and A for NITERATIONS. Iteratively
    obtains the mean field potential and solves the eigenvalue equation. The ground state energy is
    returned as E_x + E_y - a*V_x*V_y. Solution is assumed to be converged after NITERATIONS.

    Parameters
    ----------
    h  2d array, the H matrix
    G  2d array, the G matrix
    S  2d array, the S matrix
    a  int, interaction strength
    niterations  int, number of maximum iterations.

    Returns
    -------
    Etot  float, the converged ground state energy.
    c1  array,  the coefficients of oscillator 1
    c2  array,  the coefficients of oscillator 2"""
    E, c = solve_eig(h, S)
    c1 = c[:, 0]
    c2 = c[:, 2]
    for iteration in range(niterations):
        avx4 = c1 @ G @ c1
        avy4 = c2 @ G @ c2
        heffx = h + a * avy4 * G
        heffy = h + a * avx4 * G
        Ex, c = solve_eig(heffx, S)
        c1 = c[:, 0]
        Ey, c = solve_eig(heffy, S)
        c2 = c[:, 0]
        Etot = Ex[0] + Ey[0] - a * avx4 * avy4
    return Etot, c1, c2


def get_phi(x, xa, alpha):
    """Returns phi(x) = exp(-\alpha(x-xa)^2)
    Parameters
    ----------
    x  2d array
    xa  int
    Returns
    -------
    2d array
    """
    return np.exp(-alpha * (x - xa) ** 2)


def get_wfn(n, c1, c2, center, alpha):
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
    f1 = 0
    f2 = 0
    #  grid points to query
    xq = np.linspace(-2.5, 2.5)
    yq = xq
    xx, yy = np.meshgrid(xq, yq)
    for j in range(n):
        f1 = f1 + c1[j] * get_phi(xx, center[j], alpha)
        f2 = f2 + c2[j] * get_phi(yy, center[j], alpha)
    return f1 * f2


if __name__ == '__main__':
    # initialize bases parameters
    alpha = 2
    deltax = 0.5
    n = 10
    K = 2 * n + 1
    center = np.unique(make_M(n, deltax)[:, 0])
    Evals = []
    avals = np.linspace(0, 0.01, 20)
    S = make_S(K, center, alpha)
    H = make_H(K, center, alpha)
    G = make_G(K, center, S, alpha)

    # ii)
    for a in avals:
        E, c1, c2 = do_scf(H, G, S, a)
        Evals.append(E)
    Epertub = 1 + 9 * avals / 16
    fig = plt.figure()

    ax = fig.add_subplot(111, xlabel="Interaction strength a",
                         ylabel="$E_0$",
                         title='Ground state energy of various interaction strength')
    ax.plot(avals, Evals, label='SCF energy')
    ax.plot(avals, Epertub, label='Pertubation approximation')
    plt.legend()
    plt.savefig('ps5_problem1ii.png')
    plt.show()

    # iii)
    avals = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 10]
    Evals = []
    for a in avals:
        E, c1, c2 = do_scf(H, G, S, a)
        Evals.append(E)
    table = pd.DataFrame({'E0': Evals}, index=avals)
    table.index.name = 'a'
    print("""iii) Energies for various a's: \n""", table)

    # iv)
    Evals_exact = [1.000553, 1.004957, 1.019445, 1.032858, 1.094749, 1.140271, 1.415094]
    print("""The more accurate result obtained from problem 4 is the following: {0}.
    Compared to our SCF solution {1}, one can see that at small values of a, the SCF solution has accuracy up to 
    the third decimal point. But at higher values of a (>0.05), the SCF solution starts
    to deviate more and more.""".format(Evals_exact, Evals))

    # v)
    E, c1, c2 = do_scf(H, G, S, a=10)
    xq = np.linspace(-2.5, 2.5)
    yq = xq
    psi = get_wfn(K, c1, c2, center, alpha)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, xlabel='x', ylabel='y',
                           title='Contour Plot of Ground State Wavefunction for a = 10')
    ax1.contourf(xq, yq, psi, levels=20, cmap='plasma')
    ax1.axis('equal')
    plt.savefig('ps5_problem1v.png')
    plt.show()
    print("""From the contour plot, while the SCF solution has the correct form near the origin, itdoes 
    not capture the behaviour of the wavefunction at larger absolute values of the grid. This is because for 
    oscillator a, the SCF solution approximates the potential of the other oscillator as a smeared out potential, 
    and vice versa; hence, the complete behaviour of the oscillators' interaction is not captured, but averaged out
    across the two.
    """)
