from ps3_problem3 import make_M, make_S, make_H, solve_eig
import numpy as np
import matplotlib.pyplot as plt

alpha = 2
def get_phi(x, y, xa, ya):
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

if __name__ == '__main__':
    #i, ii)
    # Initialize all relevant matrices
    M = make_M(9, deltax=0.5, print_=False)
    S = make_S(361, M)
    H = make_H(361, M, S)
    E, c = solve_eig(H, S)
    xa = M[:,0]
    ya = M[:,1]
    #  grid points to query
    xq = np.linspace(-3, 3)
    yq = xq
    xx, yy = np.meshgrid(xq, yq)
    psi_0 = 0
    #  compute the linear combination of psi from the coeffs in c. get_phi returns a 2d array of
    #phi(x,y) evaluated at every grid point with a given xa, ya, and coeff.
    for i in range(H.shape[0]):
        psi_0 = psi_0 + c[i, 0] * get_phi(xx, yy, xa[i], ya[i])
    #  plots the contour
    plt.contourf(xx, yy, psi_0, levels=20, cmap='plasma')
    plt.title('Contour plot of the ground state wavefunction')
    plt.axis('equal')
    plt.savefig('ps3_problem5ii.png')
    plt.show()

    #iii)
    psi_1 = 0
    for i in range(H.shape[0]):
        psi_1 = psi_1 + c[i, 1] * get_phi(xx, yy, xa[i], ya[i])
    psi_2 = 0
    for i in range(H.shape[0]):
        psi_2 = psi_2 + c[i, 2] * get_phi(xx, yy, xa[i], ya[i])
    fig1 = plt.figure()
    plt.contourf(xx, yy, psi_1, levels=20, cmap='plasma')
    plt.title('Contour plot of the degenerate (i) second energy level wavefunction')
    plt.axis('equal')
    plt.savefig('ps3_problem5iiia.png')
    plt.show()
    fig2 = plt.figure()
    plt.contourf(xx, yy, psi_2, levels=20, cmap='plasma')
    plt.title('Contour plot of the degenerate (ii) second energy level wavefunction')
    plt.axis('equal')
    plt.savefig('ps3_problem5iiib.png')
    plt.show()

    #iv)
    psi_3 = 0
    for i in range(H.shape[0]):
        psi_3 = psi_3 + c[i, 3] * get_phi(xx, yy, xa[i], ya[i])
    psi_4 = 0
    for i in range(H.shape[0]):
        psi_4 = psi_4 - c[i, 4] * get_phi(xx, yy, xa[i], ya[i])
    psi_5 = 0
    for i in range(H.shape[0]):
        psi_5 = psi_5 + c[i, 5] * get_phi(xx, yy, xa[i], ya[i])
    fig3 = plt.figure()
    plt.contourf(xx, yy, psi_3, levels=20, cmap='plasma')
    plt.title('Contour plot of the degenerate (i) third energy level wavefunction')
    plt.axis('equal')
    plt.savefig('ps3_problem5iva.png')
    plt.show()
    fig4 = plt.figure()
    ax4 = fig4.gca(projection='3d')
    ax4.plot_surface(xx, yy, psi_4, cmap='plasma')
    ax4.view_init(30, 0)
   #plt.contourf(xx, yy, psi_4, levels=20, cmap='plasma')
    #plt.title('Contour plot of the degenerate (ii) third energy level wavefunction')
    #plt.axis('equal')
    plt.savefig('ps3_problem5ivb.png')
    plt.show()
    fig5 = plt.figure()
    plt.contourf(xx, yy, psi_5, levels=20, cmap='plasma')
    plt.title('Contour plot of the degenerate (iii) third energy level wavefunction')
    plt.axis('equal')
    plt.savefig('ps3_problem5ivc.png')
    plt.show()



