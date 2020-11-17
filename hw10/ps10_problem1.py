import numpy as np
import matplotlib.pyplot as plt
from oscillators_Monte_Carlo import histogram

def do_monte_carlo(M, E, steps):
    '''Does a monte carlo procedure for number of oscillators M, total energy E, and up to STEPS.'''
    h = histogram(limits=[-0.5,E+0.5],binwidth=1)

    n_quanta = np.zeros(M).astype(int) # energies of the particles
    n_quanta[0] = E # all energies at particle 1 initially
    n_trajectory = np.zeros(steps) # trajectory array
    for step in range(steps):
        # random donor and acceptor
        donor = int(M * np.random.rand())
        acceptor = int(M * np.random.rand())

        if n_quanta[donor] > 0:
            n_quanta[donor] -= 1 # remove energy from donor
            n_quanta[acceptor] += 1 # increase energy for acceptor

        n_trajectory[step] = n_quanta[0] # store the energy change of particle 1
        h.add_sample(n_quanta[0])
    return n_trajectory


if __name__ == '__main__':
    np.random.seed(42) # keep random output constant
    #i)
    print("""i) See oscillators_Monte_Carlo.py for comments.""")

    #ii)
    M = 10
    E = 30
    steps = 10000
    trajectory = do_monte_carlo(M, E, steps)
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='Steps', ylabel='Energy', title='Energy of oscillator 1 vs steps')
    ax.plot(np.arange(steps), trajectory)
    plt.savefig('ps10_problem1ii.png')
    plt.show()

    #iii)
    M = 10
    E = 30
    steps = 10000
    traj_arr = [] # store the trajectories for later use
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, xlabel='Steps', ylabel='Energy', title='Energy of oscillator 1 iterated 100 times')
    for i in range(100):
        trajectory = do_monte_carlo(M, E, steps)
        traj_arr.append(trajectory)
        ax1.plot(np.arange(steps), trajectory)
    plt.savefig('ps10_problem1iii.png')
    plt.show()

    #iv)
    mean_arr = np.mean(np.array(traj_arr), axis=0) # average of trajectories for fixed step S
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, xlabel='Steps', ylabel='Average energy of a given step',
                           title='Average energy change over steps')
    ax2.plot(np.arange(steps), mean_arr)
    plt.savefig('ps10_problem1iv.png')
    plt.show()
    relax_ind = np.argwhere(mean_arr<10)[0][0] # get index
    print("""iv) The first occurrence of step that has average energy less than 10 is {0}""".format(relax_ind))

    #v)
    M = 20
    E = 60
    steps = 10000
    trajectory = do_monte_carlo(M, E, steps)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, xlabel='Steps', ylabel='Energy', title='Energy of oscillator 1 vs steps with M=20, '
                                                                       'E=60')
    ax3.plot(np.arange(steps), trajectory)
    plt.savefig('ps10_problem1va.png')
    plt.show()
    traj_arr = []
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111, xlabel='Steps', ylabel='Energy',
                           title='Energy of oscillator 1 iterated 100 times with M=20, E=60')
    for i in range(100):
        trajectory = do_monte_carlo(M, E, steps)
        traj_arr.append(trajectory)
        ax4.plot(np.arange(steps), trajectory)
    plt.savefig('ps10_problem1vb.png')
    plt.show()
    mean_arr = np.mean(np.array(traj_arr), axis=0)
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111, xlabel='Steps', ylabel='Average energy of a given step',
                           title='Average energy change over steps for system M=20, E=60')
    ax5.plot(np.arange(steps), mean_arr)
    plt.savefig('ps10_problem1vc.png')
    plt.show()
    relax_ind = np.argwhere(mean_arr < 10)[0][0]
    print("""v) The first occurrence of step that has average energy less than 10 is {0}""".format(relax_ind))

    #vi)
    print("""vi) It takes more steps for the system with M=20, E=60 to reach the first occurrence where the average
    energy is below 10. From the energy perspective, since its initial energy is higher, it would naturally take 
    more step to dissipate to below E=10 threshold compared to starting at E=30. Furthermore, with more oscillators,
    the probability that the first oscillator will be affected is naturally lower than for M=10.""")


