# import packages for basic math, plotting, linear algebra, etc.
from numpy import *
from numpy.linalg import *
from matplotlib.pyplot import *
from ps5_problem3 import make_HSG
from ps5_problem2 import make_M


def eigSinvH(S, H):
    SinvH = inv(S) @ H
    E, U = eig(SinvH)

    order = argsort(E)
    c = zeros((K, K))
    for i in range(K):
        c[:, i] = U[:, order[i]]
        c[:, i] = c[:, i] / sqrt(c[:, i] @ S @ c[:, i])

    E = sort(E)

    return E, c


alpha = 2
deltax = 0.5

n = 10
K = 2 * n + 1

center = arange(-n * deltax, (n + 1) * deltax, deltax)

S = zeros((K, K))
h = zeros((K, K))
G = zeros((K, K))

for A in range(K):
    xA = center[A]
    for B in range(K):
        xB = center[B]
        #  populate overlap matrix element
        S[A, B] = sqrt(0.5 * pi / alpha) * exp(-0.5 * alpha * (xA - xB) ** 2)
        #  populate hamiltonian matrix element
        h[A, B] = 0.5 * S[A, B] * (alpha - alpha ** 2 * (xA - xB) ** 2 + \
                                   0.25 * (1 / alpha + (xA + xB) ** 2))
        #  populate g matrix element
        G[A, B] = S[A, B] * (3 / (16 * alpha ** 2) + \
                             (3 / (8 * alpha)) * (xA + xB) ** 2 + \
                             (1 / 16) * (xA + xB) ** 4)

# initial guess for the eigenvalue and eigenvector
E, c = eigSinvH(S, h)
c1 = c[:, 0]
c2 = c[:, 1]


#  initial guess for the density matrix

def do_scf(h, S, G, a, c1i, c2i, niterations=100):
    """Does the scf procedure

    Parameters
    ----------
    h  initial h matrix
    S  initial S matrix
    G  initial G matrix
    a  interacting strength
    c1i  initial coefficient guess
    c2i  initial coefficient guess
    niterations  number of iterations

    Returns
    -------
    Etot  the converged total energy
    c1  the coefficient for oscillator1's wfn
    c2  the coefficient for oscillator2's wfn
    """
    P = zeros((K, K))
    c1 = c1i
    c2 = c2i
    #  set limit of SCF iterations, assume to converge
    for iteration in range(niterations):
        for D in range(K):
            for E in range(K):
                P[D, E] = c2[D] * c2[E] + c1[D] * c1[E]
        # P = outer(c1,c1) + outer(c2,c2)
        #  construct the "fock" matrix
        heff = h + a * (trace(P @ G) * G - G @ P @ G)
        # update the eigenvalue and eigenvectors
        E, c = eigSinvH(S, heff)
        c1 = c[:, 0]
        c2 = c[:, 1]
        e1 = E[0]
        e2 = E[1]
        # prints the total energy of the system
        Etot = 0.5 * (e1 + e2 + trace(P @ h))
    return Etot, c1, c2


# Exact energy for a=1 is about 2.438
# iii)
Etot_vals = []
a_vals = np.arange(0, 101)
for a in a_vals:
    E_tot, _, _ = do_scf(h, S, G, a, c1, c2)
    Etot_vals.append(E_tot)

fig = figure()
ax = fig.add_subplot(111, xlabel='interacting strength', ylabel='Approx Energy (ha)',
                     title='SCF energy vs interacting strength')
plot(a_vals, Etot_vals)
savefig('ps6_problem1iii.png')
show()

# iv)
alpha = 2
deltax = 0.5
n = 10
K = n * (2 * n + 1)
center = make_M(n, deltax)
H, S, G = make_HSG(K, center, alpha)
Evals_bse = []
for a in a_vals:
    Hnew = H + a * G
    E, _ = eigSinvH(S, Hnew)
    Evals_bse.append(E[0])
fig1 = figure()
ax1 = fig1.add_subplot(111, xlabel='interacting strength', ylabel='Approx Energy (ha)',
                       title='SCF energy vs BSE energy comparison')
plot(a_vals, Etot_vals, label='SCF solution')
plot(a_vals, Evals_bse, label='Basis set expansion solution')
legend()
savefig('ps6_problem1iv.png')
show()
print("""iv) Our SCF approximation performs well at low interacting strength, and quickly diverge away
 from the exact solution when interaction strength goes above single digit. The SCF solution is consistently
 larger than the exact solution, which is expected because the procedure should be variational.""")

# v)
alpha = 2
deltax = 0.5

n = 10
K = 2 * n + 1

center = arange(-n * deltax, (n + 1) * deltax, deltax)

S = zeros((K, K))
h = zeros((K, K))
G = zeros((K, K))

for A in range(K):
    xA = center[A]
    for B in range(K):
        xB = center[B]
        #  populate overlap matrix element
        S[A, B] = sqrt(0.5 * pi / alpha) * exp(-0.5 * alpha * (xA - xB) ** 2)
        #  populate hamiltonian matrix element
        h[A, B] = 0.5 * S[A, B] * (alpha - alpha ** 2 * (xA - xB) ** 2 + \
                                   0.25 * (1 / alpha + (xA + xB) ** 2))
        #  populate g matrix element
        G[A, B] = S[A, B] * (3 / (16 * alpha ** 2) + \
                             (3 / (8 * alpha)) * (xA + xB) ** 2 + \
                             (1 / 16) * (xA + xB) ** 4)

# initial guess for the eigenvalue and eigenvector
E, c = eigSinvH(S, h)
c1i = c[:, 0]
c2i = c[:, 1]
E, c1, c2 = do_scf(h, S, G, 10, c1i, c2i)

# set plotting parameters
xvals = arange(-4, 4, 0.1)
yvals = xvals
nvals = size(xvals)

chi1 = 0 * xvals
chi2 = 0 * yvals

# begin allocating the individual wavefunctions for each oscillator
for A in range(K):
    chi1 += c1[A] * exp(-alpha * (xvals - center[A]) ** 2)
    chi2 += c2[A] * exp(-alpha * (xvals - center[A]) ** 2)
plot(xvals, chi1)
plot(yvals, chi2)

clf()
psi = zeros((nvals, nvals))
# calculate the total wavefunction of the combined system.
for i in range(nvals):
    x = xvals[i]
    for j in range(nvals):
        y = yvals[j]

        psi[i, j] = (1 / sqrt(2)) * (chi1[i] * chi2[j] \
                                     - chi2[i] * chi1[j])

clf()
contourf(xvals, yvals, psi, levels=20, cmap=cm.plasma)
axis('equal')
xlabel('x')
ylabel('y')
title('Contour Plot of the SCF Ground State Wavefunction, a = 10')
savefig('ps6_problem1v.png')
show()
print("""v) Compared to the exact solution plotted in the last problem set, there is much
less variation of the contour at values further away from the origin. Near the origin, however,
the two plots show similar shape.""")
