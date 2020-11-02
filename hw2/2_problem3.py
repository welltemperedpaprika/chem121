import numpy as np
import matplotlib.pyplot as plt

alpha = np.linspace(1/4, 1, 500)  #set points to query
delta_x = np.linspace(0, 1, 500)[1:-1]  #set points to query

if __name__ == '__main__':
    #i)
    y1 = 1/2*(alpha + 1/(4*alpha))
    plt.plot(alpha, y1)
    plt.xlabel('$\\alpha$')
    plt.ylabel('Energy')
    plt.title('Finding optimal $\\alpha$ wrt/ energy')
    plt.savefig('2_problem3_1.png')
    plt.show()

    #ii)
    print("""ii) The ground state energy is {0}, and according to the 
          variational principle, the true ground state 
          is lower than that.""".format(np.min(y1)))

    #iii)
    y2 = (1 + delta_x**2 - np.exp(-delta_x**2) * (1 - delta_x**2)) \
             / (2 * (1 - np.exp(-delta_x**2)))
    plt.plot(delta_x, y2)
    plt.xlabel('$\Delta x$')
    plt.ylabel('Energy')
    plt.title('Finding optimal $\Delta x$ wrt/ energy')
    plt.savefig('2_problem3_2.png')
    plt.show()

    #iv)
    print("""iv) It seems that for delta x close to 0, the energy is
     at a minimum with value {0}""".format(min(y2)))

    #v)
    print("""The trial wave function is an odd function.
    But the solution to the QHO has ground state with even symmetry. Hence,
    by parity, the trial wavefunction is a poor approximate to the ground state.""")