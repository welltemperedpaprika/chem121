import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import pandas as pd

class histogram():
    ## a histogram object
    def __init__(self, limits, binwidth):
        self.limits = limits
        self.binwidth = binwidth
        self.vals = np.arange(self.limits[0] + self.binwidth / 2, self.limits[1], self.binwidth)
        self.histo = 0 * self.vals
        self.N_samples = 0

    def add_sample(self, dat):
        self.N_samples += 1
        if dat > self.limits[0] and dat < self.limits[1]:
            bin_index = int((dat - self.limits[0]) / self.binwidth)
            self.histo[bin_index] += 1

    def normalize(self):
        self.histo = self.histo / (self.N_samples * self.binwidth)

    def barplot(self):
        plt.bar(self.vals, self.histo, width=0.95 * self.binwidth, color='k')

    def lineplot(self):
        plt.plot(self.vals, self.histo)


def plot_circle(center, radius):
    '''plots a circle at CENTER with RADIUS'''
    npoints = 100
    theta = np.arange(0, 2 * np.pi + 1e-7, 2 * np.pi / npoints)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    plt.plot(x, y, 'k', linewidth=2)

@jit(nopython=True)
def init_config(N):
    '''initialize configuration'''
    r = np.zeros((N, 2))  # array of x, y positions of particles
    n_side = int(np.sqrt(N) + 0.99)  # start with a square lattice
    count = 0
    # allocate the position arrays
    for row in range(n_side):
        for column in range(n_side):
            if count < N:
                r[count, :] = [row, column]
                count += 1

    return r


def draw_config(r, box_length):
    '''draw the configuration on a graph'''
    N = r.shape[0]
    plt.clf()
    for i in range(N):
        r[i, :] -= box_length * np.floor(r[i, :] / box_length + 0.5)
        plot_circle(r[i, :], 0.5)
    plt.axis('equal')
    plt.gca().set_adjustable("box")
    view_scale = 1.1 * box_length / 2
    plt.xlim(-view_scale, view_scale)
    plt.ylim(-view_scale, view_scale)

    boundary_x = box_length * (np.array([0, 1, 1, 0, 0]) - 0.5)
    boundary_y = box_length * (np.array([0, 0, 1, 1, 0]) - 0.5)
    plt.plot(boundary_x, boundary_y)

    plt.pause(0.01)

@jit(nopython=True)
def MC_sweep(r, box_length, max_hop_length, T):
    N = r.shape[0]
    r_cut = box_length / 2
    r_cut_squared = r_cut ** 2
    for step in range(N):
        i_trial = int(np.random.rand()*N) # set trial random particle
        r_trial = r[i_trial,:] + max_hop_length * (np.random.rand(2) - 0.5)
        delta_U = 0 # begin calculating change in energy
        for j in range(N):
            if j != i_trial:
                dr_trial = r_trial - r[j,:] # vector distance between ij
                dr_trial -= box_length * np.floor(dr_trial/box_length + 0.5) # account for periodicity
                dr2_trial = dr_trial @ dr_trial
                dr2_trial = min(dr2_trial,r_cut_squared)

                dr = r[i_trial,:] - r[j,:]
                dr -= box_length * np.floor(dr/box_length + 0.5)
                dr2 = dr @ dr
                dr2 = min(dr2,r_cut_squared)

                delta_U += 4*( dr2_trial**(-6) - dr2_trial**(-3) ) \
                    - 4*( dr2**(-6) - dr2**(-3) )
        if np.random.rand() < np.exp(-delta_U / T): # selection rule
            r[i_trial,:] = r_trial
    return r

@jit(nopython=True)
def get_U(r, box_length):
    '''Calculates the total potential energy given a configuration R and BOX_LENGTH.'''
    result = 0
    r_cut = box_length / 2
    r_cut_squared = r_cut ** 2
    N = r.shape[0]
    for i in range(N):
        for j in range(N):
            if j > i: #iterate over unique ij pairs
                dr = r[i, :] - r[j, :]
                dr -= box_length * np.floor(dr / box_length + 0.5)
                dr2 = dr @ dr
                dr2 = min(dr2, r_cut_squared)
                result += 4 * (dr2 ** (-6) - dr2 ** (-3)) - (4 * (r_cut_squared ** (-6) - r_cut_squared ** (-3)))
    return result


def get_statistics(N, density, max_hop_length, N_sweeps, T):
    '''compact function to calculate mean U per particle and variance of U per particle.'''
    box_length = np.sqrt(N / density)
    r = init_config(N)
    total_U_arr = []
    for sweep in np.arange(N_sweeps):
        r = MC_sweep(r, box_length, max_hop_length, T)
        if sweep % 100 == 0:
            total_U_arr.append(get_U(r, box_length))
    total_U_arr = np.array(total_U_arr)[1:]
    mean_U = np.mean(total_U_arr)
    var_U = np.mean(total_U_arr ** 2) - mean_U ** 2
    return mean_U/N, var_U/N



if __name__ == '__main__':
    s = 0 # set initial seed
    #i)
    print("""i) See LJ_periodic.py for comments. Also see consoleText.pdf for output for part iii) to vi) if taking 
    too long.""")

    #ii)
    N = 36  # set number of particles
    T = 0.8  # set low temperature
    density = 0.7
    box_length = np.sqrt(N / density)
    max_hop_length = 0.5
    r = init_config(N)
    N_sweeps = 10000
    N_sweeps_interval = np.arange(N_sweeps, step=100)
    total_U_arr = []
    for sweep in np.arange(N_sweeps):
        r = MC_sweep(r, box_length, max_hop_length, T)
        if sweep % 100 == 0:  # draw every 100 sweeps
            total_U_arr.append(get_U(r, box_length))
    fig = plt.figure()
    ax = fig.add_subplot(111, title='total LJ energy over sweeps', xlabel='sweep', ylabel='$U_{LJ}$')
    ax.plot(N_sweeps_interval, total_U_arr)
    plt.savefig('ps11_problem1i.png')
    plt.show()

    #iii)
    print('''iii) From the graph, it appears that the system reaches equilibrium fast at the 100th sweep.''')

    #iv)
    N_sweeps = 100000
    mean_U, var_U = get_statistics(N, density, max_hop_length, N_sweeps, T)
    print('''iv) The average U per particle is {0}, and the variance of U per particle is {1}'''.format(mean_U,
                                                                                                        var_U))
    #v)
    mean_U_arr = [mean_U]
    var_U_arr = [var_U]
    T_arr = [0.8, 0.75, 0.7, 0.65]
    for t in T_arr[1::]:
        mean_U_temp, var_U_temp = get_statistics(N, density, max_hop_length, N_sweeps, t)
        mean_U_arr.append(mean_U_temp)
        var_U_arr.append(var_U_temp)
    table = pd.DataFrame({'T': T_arr, 'avg_U_per_particle': mean_U_arr, 'var_U_per_particle': var_U_arr})
    print('''v) See table
    {0}'''.format(table))

    #vi)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, title='average energy per particle vs T', xlabel='T',
                           ylabel='$\langle U_{LJ} \\rangle$')
    ax1.scatter(T_arr, mean_U_arr)
    plt.savefig('ps11_problem1v.png')
    plt.show()
    print('vi) From the plot, the derivative is about {0}.'.format((mean_U_arr[-1]-mean_U_arr[-2])/
                                                                   (T_arr[-1]-T_arr[-2])))

    #vii)
    c_config_calc = np.array(var_U_arr) / (np.array(T_arr) ** 2)
    print('''vii) From calculated variances, the c_configs are about {0}, and compared to the value eyeballed from
    the derivative, the numerical values are close for temperature between 0.75 and 0.7.'''.format(c_config_calc))

