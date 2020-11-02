import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, pi, exp
from numpy.linalg import inv, eig
import pandas as pd

def get_s_ab(a, b, alpha=0.5):
    return sqrt(pi/(2*alpha)) * exp(-alpha/2 * (a - b)**2)

def get_g_ab(a, b, alpha=0.5):
    s_ab = get_s_ab(a, b, alpha)
    return s_ab * (3/(16*alpha**2) + 3/(8*alpha)*(a + b)**2 + 1/16*(a + b)**4)

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
    query = np.arange(-n*deltax, n*deltax + deltax, deltax)
    xx, yy = np.meshgrid(query, query)
    c1 = xx.flatten('F')
    c2 = yy.flatten('F')
    M = np.column_stack((c1, c2))
    return M

def make_S(n, M, alpha):
    """makes the overlap matrix. Implementation utilizes numpy's element wise operation. rows and cols contain
    the indices of S_{AB} in the form of row: [{0}_n, {1}_n, ...], col: [{0, 1..,n}_n] such that they extract
    x and y in the form of x[rows]:[{x0}_n, {x1}_n...] etc. The result is then reshaped to an nxn matrix.

    Parameters
    ----------
    n  int, dimension of resulting S matrix
    alpha  int, the width factor
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

def make_G(n, M, S, alpha):
    """makes the G matrix. Implementation utilizes numpy's element wise operation. rows and cols contain
    the indices of H_{AB} in the form of row: [{0}_n, {1}_n, ...], col: [{0, 1..,n}_n] such that they extract
    x and y in the form of x[rows]:[{x0}_n, {x1}_n...] etc. The result is then reshaped to an nxn matrix.

    Parameters
    ----------
    n  int, dimension of resulting H matrix
    alpha  int, the width factor
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
    G = S * (3/(16*alpha**2) + 3/(8*alpha)*(x[rows] + x[cols])**2 + 1/16*(x[rows] + x[cols])**4)
    G = G * (3/(16*alpha**2) + 3/(8*alpha)*(y[rows] + y[cols])**2 + 1/16*(y[rows] + y[cols])**4)
    G = np.reshape(G, (n, n))
    return G

def make_H(n, M, a, alpha):
    """makes the hamiltonian matrix. Implementation utilizes numpy's element wise operation. rows and cols contain
    the indices of H_{AB} in the form of row: [{0}_n, {1}_n, ...], col: [{0, 1..,n}_n] such that they extract
    x and y in the form of x[rows]:[{x0}_n, {x1}_n...] etc. The result is then reshaped to an nxn matrix.
    Parameters
    ----------
    n  int, dimension of resulting H matrix
    a  int, the interaction factor
    alpha  int, the width factor
    M  2d array, an nx2 array where the first column is x indices, and second col is y indices.

    Returns
    -------
    2d array
    """
    S = make_S(n, M, alpha)
    G = make_G(n, M, S, alpha)
    indices = np.arange(n)
    rows, cols = np.meshgrid(indices, indices)
    rows = rows.flatten('F')
    cols = cols.flatten('F')
    x = M[:, 0]
    y = M[:, 1]
    S = S.flatten()
    G = G.flatten()
    H = 0.5 * S * (2*alpha + 1/(2*alpha) - alpha**2*(x[rows] - x[cols])**2 - alpha**2*(y[rows] - y[cols])**2
                            + 0.25*(x[rows] + x[cols])**2 + 0.25*(y[rows] + y[cols])**2)
    H = H + a * G
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

def get_phi(x, y, xa, ya, alpha):
    """Returns phi(x, y) = exp(-\alpha(x-xa)^2 - \alpha(y-ya)^2)
    Parameters
    ----------
    x  2d array
    y  2d array
    xa  int
    ya  int
    Returns
    -------
    2d array
    """
    return np.exp(-alpha*(x - xa)**2 - alpha*(y - ya)**2)

def get_wfn(n, c, M, alpha, i=0):
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
    xa = M[:, 0]
    ya = M[:, 1]
    #  grid points to query
    xq = np.linspace(-2.5, 2.5)
    yq = xq
    xx, yy = np.meshgrid(xq, yq)
    for j in range(n):
        psi = psi + c[j, i] * get_phi(xx, yy, xa[j], ya[j], alpha)
    return psi

if __name__ == '__main__':
    #i)
    xq = np.linspace(-5, 5)
    xx, yy = np.meshgrid(xq, xq)
    z = xx**4 * yy**4
    plt.contourf(xx, yy, z)
    plt.title('Contour plot of $z=x^4y^4$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('ps4_problem1a.png')
    plt.show()

    #ii)
    print("""ii) The matrix element of 
    \langle phi_A \otimes phi_B | x^4y^4 | \phi_A \otimes \phiB \\rangle
    is the product of \langle \phi_A | x^4 | \phi_A \\rangle and \langle \phi_B | y^4 | \phi_B \\rangle. 
    Explicitly, \langle phi_A \otimes phi_B | x^4y^4 | \phi_A \otimes \phiB \\rangle = 
    s(xA, xB)*s(yA, yB)*(3/(16\\alpha^2) + 3/(8\\alpha)(xA + xB)^2 + 1/16(xA+xB)^4)*(3/(16\\alpha^2)
     + 3/(8\\alpha)(yA + xB)^2 + 1/16(yA+yB)^4)
    e.g. when alpha = 0.5, xa, xb, ya, yb = 0, get_g_ab(0,0)*get_g_ab(0,0) =""", get_g_ab(0,0)*get_g_ab(0,0))

    #iii)
    print("""iii) See line 59 for construction of matrix G.""")

    #iv)
    print("""iv) See line 86 for construction of matrix H.""")

    #v)
    M = make_M(7, deltax=0.5)
    S = make_S(225, M, alpha=2)
    H = make_H(225, M, a=0.005, alpha=2)
    E, c = solve_eig(H, S)
    print("""v) The ground state energy is""", E[0])

    #vi)
    E_0 = []
    a_vals = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 10]
    for a_i in a_vals:
        H = make_H(225, M, a_i, alpha=2)
        E, c = solve_eig(H, S)
        E_0.append(E[0])
    table = pd.DataFrame(E_0, index=a_vals, columns=['E_0'])
    print("vi)\n", table)

    #vii)
    E_0pt = []
    for a_i in a_vals:
        E_0pt.append(1 + 9/16*a_i)
    table['E_0pt'] =  E_0pt
    table['error'] = (table['E_0pt'] - table['E_0'])/table['E_0']
    print("vii)\n", table)
    print("From a=0.001 to a=0.05, the error is within 1%")

    #viii)
    H = make_H(225, M, a=1, alpha=2)
    E, c = solve_eig(H, S)
    xq = np.linspace(-2.5, 2.5)
    xx, yy = np.meshgrid(xq, xq)
    psi = get_wfn(225, c, M, alpha=2, i=0)
    plt.figure(1)
    plt.contourf(xx, yy, psi)
    plt.title('Contour plot of ground state wfn for a=1')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('ps4_problem1viii.png')
    plt.show()

    #ix)
    a_vals = [0.001, 0.01, 0.05, 0.1, 0.5, 10]
    for count, a_i in enumerate(a_vals):
        H = make_H(225, M, a_i, alpha=2)
        E, c = solve_eig(H, S)
        psi = get_wfn(225, c, M, alpha=2, i=0)
        plt.contourf(xx, yy, psi)
        plt.title('Contour plot of ground state wfn for a={0}'.format(a_i))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('ps4_problem1ix{0}.png'.format(count))
        plt.show()
    print("""ix) It seems that for a=0.05, the structure for the ground state has changed significantly, where the
    ground state has changed sign. This is consistent with pertubation theory's range of validity.""")