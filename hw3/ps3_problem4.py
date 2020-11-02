from ps3_problem3 import make_M, make_S, make_H, solve_eig
import numpy as np
import matplotlib.pyplot as plt

def harmonic_oscillator_2d_scan_energy(ki=3, kf=9, nlowest=45):
    """plots the nlowest energy levels as a function of number of basis functions by combining previously written
    functions together.

    Parameters
    ----------
    ki  int, number of basis functions from
    kf  int, number of basis functions to
    nlowest  int, determines the maximum of energy levels to plot to

    Returns
    -------
    None
    """
    nb = np.arange(ki, kf+1)  #number of basis functions as a list
    kval = (2*nb + 1)**2   #size of each relevant matrix for a number of basis functions
    Earray = []
    for i in range(len(nb)):
        M = make_M(nb[i], deltax=0.5, print_=False)
        S = make_S(kval[i], M)
        H = make_H(kval[i], M, S)
        E, c = solve_eig(H, S)
        Earray.append(np.real(E[0:nlowest]))
    Earray = np.row_stack(Earray)
    for i in range(nlowest):
        plt.plot(kval, Earray[:, i])
    plt.title("energy levels as a function of # of basis functions")
    plt.xlabel('# of basis functions', fontsize=14)
    plt.ylabel('energy', fontsize=14)
    plt.savefig('ps3_problem4.png')
    plt.show()

if __name__ == '__main__':
    #i)
    M = make_M(3, deltax=0.5, print_=False)
    S = make_S(49, M)
    H = make_H(49, M, S)
    E, c = solve_eig(H, S)
    print("i) The first 45 energies are:", np.real(E)[0:45])

    #ii)
    print("""ii) The energies are not exact in both the values and degeneracies compared
          to the exact solution. e.g. the third energy levels and beyond do not have true
          degeneracies, and the multiplicity of the degeneracies also fall apart at E = 5.""")

    #iii)
    print("iii) See harmonic_oscillator_2d_scan_energy at line 5 for code.")

    #iv)
    harmonic_oscillator_2d_scan_energy()

    #v)
    print("""v) From the graph, it looks like around K=225 the energy values start to converge
    for each energy level.""")

    #vi)
    print("""vi) For the 2d case, the amount of basis functions required to converge is about 
    the square of that in the 1d case. This is because in 1d, K = 2n + 1, 
    whereas in 2d, K = (2n + 1)^2.""")

    #vii)
    print("""vii) By the trend, the number of basis functions would be the cube of that in the 
    1d case. This is because for a given n, there is three degrees of freedom (x, y, z), and each 
    x, y, or z can take in 2n+1 values - hence (2n+1)^3. Increasing dimensionality scales the number
    of basis functions required as O(n^d), where d is the dimension.""")

    #viii)
    print("""viii) The exponential growth is clearly unfavoured since computers have limited computing
    power and memory limit. For a system requiring 10 basis funcs for 1 electron, extending to
    30 electrons would require 10^30 basis functions.""")