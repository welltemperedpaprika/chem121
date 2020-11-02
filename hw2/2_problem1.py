import numpy as np
from scipy.constants import pi
import matplotlib.pyplot as plt
#iii)

# L = 1


xq = np.linspace(0, 1, 500)  #set x points to query
A = 10  #Amplitude of wavefunction

def E_n(n):
    """
    Get the nth energy level of particle in a box.
    Parameters
    ----------
    n: int

    Returns
    -------
    int
    """
    return n**2 * pi**2

def psi_n(n):
    """
    Get the nth energy wavefunction of particle in a box.
    Parameters
    ----------
    n: int

    Returns
    -------
    array
    """
    return A * np.sin(n * pi * xq)

if __name__ == '__main__':
    #begin plotting
    plt.plot(xq, psi_n(1) + E_n(1), color='orange', label='n=1')
    plt.plot(xq, psi_n(2) + E_n(2), 'm', label='n=2')
    plt.plot(xq, psi_n(3) + E_n(3), 'y', label='n=3')
    plt.plot(xq, psi_n(4) + E_n(4), 'k', label='n=4')
    plt.hlines((0, E_n(1), E_n(2), E_n(3), E_n(4)), 0, 1, linestyles='dashed')
    plt.vlines((0,1), 0, 200, linestyles='solid')
    plt.xlabel('x')
    plt.ylabel('Energy')
    plt.legend()
    plt.title('4 Lowest energy eigenstates of particle in a box')
    plt.savefig('2_problem1.png')
    plt.show()
