import numpy as np
from oscillators_Monte_Carlo_v3 import histogram
from numba import jit
import matplotlib.pyplot as plt
from timeit import default_timer as timer

@jit(nopython=True)
def MC_sweep(M, n_quanta):
    '''Does a monte carlo sweep for number of particles M and total energy n_quanta'''
    for step in range(M):
        donor = int(M * np.random.rand())
        acceptor = int(M * np.random.rand())

        if n_quanta[donor] > 0:
            n_quanta[donor] -= 1
            n_quanta[acceptor] += 1

    return n_quanta

def do_monte_carlo(EpM, M, N_sweeps):
    '''Does a monte carlo for number of particles M, energy per particle EpM, and number of sweeps N_sweeps'''
    E = M * EpM
    h = histogram(limits=[-0.5, E + 0.5], binwidth=1)
    n_quanta = np.ones(M).astype(int) * EpM
    n_trajectory = np.zeros(N_sweeps) # the trajectory of the first oscillator
    for sweep in range(N_sweeps):
        # for step in range(M):
        #     donor = int(M * rand())
        #     acceptor = int(M * rand())
        #
        #     if n_quanta[donor] > 0:
        #         n_quanta[donor] -= 1
        #         n_quanta[acceptor] += 1
        n_quanta = MC_sweep(M, n_quanta)
        n_trajectory[sweep] = n_quanta[0]
        h.add_sample(n_quanta[0])
    return n_trajectory

def get_corrlation(n_trajectory, tau, EpM):
    '''calculates the average correlation between trajectories-EpM at tau apart. Utilizes element wise multiplication
    between tau shifted arrays.'''
    return np.mean(np.multiply(n_trajectory[0:-tau]-EpM, n_trajectory[tau::]-EpM))

if __name__ == '__main__':
    np.random.seed(42) # keep random output constant
    #i)
    M = 10
    EpM = 3
    N_sweeps = 10000
    n_trajectory = do_monte_carlo(EpM, M, N_sweeps)
    n1_avg = np.mean(n_trajectory)
    n1_meansq = np.mean(n_trajectory**2)
    n1_meansq_exact = EpM**2 + (M-1)/(M+1)*EpM*(1+EpM)
    print("""i) The calculated average number of quanta is {0}, and the theoretical value is {1}.
    The calculated mean square number of quanta is {2}, and the theoretical value is {2}. They agree pretty 
    well""".format(n1_avg, EpM, n1_meansq, n1_meansq_exact))
    #ii)
    EpM = 3
    N_sweeps = 1000
    for m in [10, 100, 10000]:
        n_trajectory = do_monte_carlo(EpM, m, N_sweeps)
        delta_t = n_trajectory - EpM
        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel='Sweeps', ylabel='$\delta n1$', title='Deviation of trajectory from average '
                                                                              'for M={0}'.format(m))
        ax.plot(np.arange(N_sweeps), delta_t)
        plt.savefig('ps10_problemiiM={0}'.format(m))
        plt.show()
        plt.clf()
    print("""ii) It seems that regardless of M, the scale of deviation can be large. But in general, they appear to 
    have similar scale of deviation. Similarly, the duration of deviation also seem to be consistent.
    This makes sense because for more particles, there is less chance that particle 1 receive many changes in energies
    in small successions.This makes sense as we are sweeping over the system as opposed to taking steps, 
    which can otherwise be biased against the system size.
    """)

    #iii)
    M = 10
    EpM = 3
    N_sweeps = 1000000
    n_trajectory = do_monte_carlo(EpM, M, N_sweeps)
    avg_corr = np.mean(np.multiply(n_trajectory[0:-1]-EpM, n_trajectory[1::]-EpM))
    print("""iii) The average correlation of neighbouring quanta for oscillator 1 is {0}""".format(avg_corr))

    #iv)
    avg_corr_100 = np.mean(np.multiply(n_trajectory[0:-100] - EpM, n_trajectory[100::] - EpM))
    print("""iv) The average correlation of quanta at 100 sweeps apart is {0}""".format(avg_corr_100))

    #v)
    M = 10
    EpM = 3
    N_sweeps = 1000000
    corr_arr = np.zeros(200)
    for c, t in enumerate(np.arange(1, 201)):
        corr_arr[c] = get_corrlation(n_trajectory, t, EpM)
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='$\\tau$', ylabel='$C_\\tau$', title='Average correlation of $\\tau$ intervals '
                                                                        'apart')
    ax.plot(np.arange(1, 201), corr_arr, label='M=10')
    plt.savefig('ps10_problem2v.png')

    #vi)
    M = 20
    n_trajectory_20 = do_monte_carlo(EpM, M, N_sweeps)
    corr_arr_20 = np.zeros(200)
    for c, t in enumerate(np.arange(1, 201)):
        corr_arr_20[c] = get_corrlation(n_trajectory_20, t, EpM)
    ax.plot(np.arange(1, 201), corr_arr_20, label='M=20')
    #######################################
    M = 30
    n_trajectory_30 = do_monte_carlo(EpM, M, N_sweeps)
    corr_arr_30 = np.zeros(200)
    for c, t in enumerate(np.arange(1, 201)):
        corr_arr_30[c] = get_corrlation(n_trajectory_30, t, EpM)
    ax.plot(np.arange(1, 201), corr_arr_30, label='M=30')
    plt.legend()
    plt.savefig('ps10_problem2vi.png')
    plt.show()
    plt.clf()

    print("""vi) As M increases, the average correlation for a fixed tau apart also increases. However, when tau
    gets larger, the average correlation quantity essentially converges regardless of system size.""")

    #vii)
    C0 = n1_meansq_exact - EpM**2
    corr_time = np.argwhere(corr_arr < (C0/np.exp(1)))[0][0] # get index
    print("""vii) The correlation time for M=10 is {0}""".format(corr_time))

    #viii)
    t = 1000
    M = 10
    n_independent = t / corr_time * M
    print("""viii) We can estimate the number of independent samples by taking the total time divide by the 
    correlation time and then multiplying by the total number of particles. For M=10 case, we have roughly {0}
    number of independent samples.""".format(n_independent))



