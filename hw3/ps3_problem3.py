import numpy as np
from numpy.linalg import eig, inv
from numpy import pi, exp, sqrt
from ps3_problem2 import make_M

alpha = 2
def make_S(n, M):
    """makes the overlap matrix. Implementation utilizes numpy's element wise operation. rows and cols contain
    the indices of S_{AB} in the form of row: [{0}_n, {1}_n, ...], col: [{0, 1..,n}_n] such that they extract
    x and y in the form of x[rows]:[{x0}_n, {x1}_n...] etc. The result is then reshaped to an nxn matrix.

    Parameters
    ----------
    n  int, dimension of resulting S matrix
    M  array, an nx2 array where the first column is x indices, and second col is y indices.

    Returns
    -------
    2d array
    """
    indices = np.arange(n)
    rows, cols = np.meshgrid(indices, indices)
    rows = rows.flatten('F')
    cols = cols.flatten('F')
    x = M[:,0]
    y = M[:,1]
    S = (pi/(2*alpha)) * exp(-(alpha/2)*(x[rows] - x[cols])**2
                             - (alpha/2)*(y[rows] - y[cols])**2)
    S = np.reshape(S, (n, n))
    return S

def make_H(n, M, S):
    """makes the hamiltonian matrix. Implementation utilizes numpy's element wise operation. rows and cols contain
        the indices of H_{AB} in the form of row: [{0}_n, {1}_n, ...], col: [{0, 1..,n}_n] such that they extract
        x and y in the form of x[rows]:[{x0}_n, {x1}_n...] etc. The result is then reshaped to an nxn matrix.

        Parameters
        ----------
        n  int, dimension of resulting H matrix
        M  2d array, an nx2 array where the first column is x indices, and second col is y indices.
        S  2d array, the S matrix
        Returns
        -------
        2d array
        """
    indices = np.arange(n)
    rows, cols = np.meshgrid(indices, indices)
    rows = rows.flatten('F')
    cols = cols.flatten('F')
    x = M[:, 0]
    y = M[:, 1]
    S = S.flatten()
    H = 0.5 * S * (2*alpha + 1/(2*alpha) - alpha**2*(x[rows] - x[cols])**2 - alpha**2*(y[rows] - y[cols])**2
                            + 0.25*(x[rows] + x[cols])**2 + 0.25*(y[rows] + y[cols])**2)
    H = np.reshape(H, (n, n))
    return H

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
        c_sorted[:,i] = c[:,sorted[i]]
        c_sorted[:,i] = c_sorted[:,i] / sqrt(c_sorted[:,i] @ S @ c_sorted[:,i])
    E = E[sorted]
    return E, c_sorted

if __name__ == '__main__':
    #i)
    print('i) Code is generalized in make_M from prob 2.'
          ' e.g. make_M(2, 0.5) gives:')
    make_M(2, 0.5)

    #ii)
    print("ii) see make_S at line 7 and make_H at line 32 for generalization.")

    #iii)
    print("iii) see solve_eig at line 59 for generalization.")
