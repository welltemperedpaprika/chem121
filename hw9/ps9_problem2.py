import numpy as np
import matplotlib.pyplot as plt
from histogram_class import histogram


def random_walk(x_init=0.5, maxhop=0.5, nsteps=1000, accept_all=True):
    ''' Does a random walk.

    Parameters
    ----------
    x_init, initial state of walker
    maxhop, maximum hop length
    nsteps, number of steps to take
    accept_all, whether or not to accept all steps

    Returns
    -------
    xtrajectory, the tracjetory of the random walk
    '''
    x = x_init
    maxhoplength = maxhop
    xtrajectory = np.zeros(nsteps)

    for step in range(nsteps):  # iterate through nsteps and allocate trajectory
        xtrial = x + maxhoplength * (np.random.rand() - 0.5)
        if accept_all: # accept all steps
            x = xtrial
            xtrajectory[step] = x
        elif xtrial >= 0 and xtrial <= 1: # accept only if the trial step is within criterion.
            x = xtrial
            xtrajectory[step] = x
    return xtrajectory

if __name__ == '__main__':
    #i)
    print('''i) See random_walk at line 6 for implementation''')

    #ii)
    steps = np.arange(0, 1000)
    xtrajectory = random_walk(accept_all=False)
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='step', ylabel='x trajectory', title='Random Monte Carlo Walk')
    ax.plot(steps, xtrajectory)
    plt.savefig('ps9_problem2ii.png')
    plt.show()
    plt.cla()

    #ii)
    long_trajectory = random_walk(nsteps=int(10e5), accept_all=True)
    h = histogram([0, 1], 0.05)
    for n in range(int(10e5)):
        h.add_sample(long_trajectory[n])
    h.normalize()
    h.barplot()
    plt.xlabel('bin')
    plt.ylabel('count')
    plt.title('Count of trajectory for nsteps=10e5 without rejecting trajectories.')
    plt.savefig('ps9_problem2iii.png')
    plt.show()
    plt.cla()

    #iii)
    long_trajectory = random_walk(nsteps=int(10e5), accept_all=False)
    h = histogram([0, 1], 0.05)
    for n in range(int(10e5)):
        h.add_sample(long_trajectory[n])
    h.normalize()
    h.barplot()
    plt.xlabel('bin')
    plt.ylabel('count')
    plt.title('Count of trajectory for nsteps=10e5 when rejecting inappropriate trajectories.')
    plt.savefig('ps9_problem2iv.png')
    plt.show()
    print('''iv) From the plot, we can see that the distribution when all moves are accepted tends towards a 
    uniform distribution, whereas when only acceptable moves are recorded, the distribution clusters around the mean.
    This means that if we neglect passage of time, all states are likely to be selected, and our Monte Carlo sampling 
    would not mean much in a physical system.''')