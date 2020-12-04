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


def draw_config(r, box_length, title):
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
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
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
        r_trial = r[i_trial,:] + max_hop_length * (np.random.rand(2) - 0.5)  # set random movement to a trial location
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
    result = 0
    r_cut = box_length / 2
    r_cut_squared = r_cut ** 2
    N = r.shape[0]
    for i in range(N):
        for j in range(N):
            if j > i:
                dr = r[i, :] - r[j, :]
                dr -= box_length * np.floor(dr / box_length + 0.5)
                dr2 = dr @ dr
                dr2 = min(dr2, r_cut_squared)
                result += 4 * (dr2 ** (-6) - dr2 ** (-3)) - 4 * (r_cut_squared ** (-6) - r_cut_squared ** (-3))
    return result

def fill_h(r, h, box_length):
    '''Fill in a histogram given configuration R, histogram H, and BOX_LENGTH'''
    N = r.shape[0]
    r_cut = box_length / 2
    for i in range(N):
        for j in range(N):
            if j > i:
                dr = r[i, :] - r[j, :]
                dr -= box_length * np.floor(dr / box_length + 0.5)
                dr = np.linalg.norm(dr)
                if dr < r_cut: # selection rule
                    h.add_sample(dr)
    return h

def plot_rdf(h, N, N_sweeps, density, title):
    '''Plot the RDF from histogram H'''
    g = h.histo / (N_sweeps * np.pi * h.vals * h.binwidth * (N - 1) * density)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, title='rdf g(r) at density {0}'.format(density), ylabel='g(r)', xlabel='r')
    ax.bar(h.vals, g, width=0.95 * h.binwidth, color='k')
    plt.savefig(title)
    plt.show()

if __name__ == '__main__':
    print('''i) Since the rdf measures the probability of finding another particle around a tagged particle,
    it is physically improbable that two particles occupy the vicinity of each other at distances smaller than their
    diameter.''')

    print('''ii) At large distances, regardless of the phase of the matter, there should be a very high chance of 
    finding another particle at distance r. Furthermore, the density of the system is related to the bulk density 
    multiplied by the rdf. At large r, the density should approach the bulk density; hence, rdf should approach 1. ''')

    #iii)
    # set params
    N = 36
    density = 0.7
    box_length = np.sqrt(N/density)
    T = 0.7
    N_sweeps = 10000
    max_hop_length = 0.4
    r_p7 = init_config(N)
    h = histogram(limits=np.array([0.1, 3.5]), binwidth=0.1)
    h = fill_h(r_p7, h, box_length)
    for sweeps in range(N_sweeps):
        r_p7 = MC_sweep(r_p7, box_length, max_hop_length, T)
        h = fill_h(r_p7, h, box_length)
    plot_rdf(h, N, N_sweeps, density, 'ps11_problem2iii.png')

    #iv)
    draw_config(r_p7, box_length,'Configuration of density=0.7')

    #vi)
    density = 0.2
    box_length = np.sqrt(N / density)
    r_p2 = init_config(N)
    h = histogram(limits=np.array([0.1, 3.5]), binwidth=0.1)
    h = fill_h(r_p2, h, box_length)
    for sweeps in range(N_sweeps):
        r_p2 = MC_sweep(r_p2, box_length, max_hop_length, T)
        h = fill_h(r_p2, h, box_length)
    plot_rdf(h, N, N_sweeps, density, 'ps11_problem2via.png')
    draw_config(r_p2, box_length, 'Configuration of density=0.2')

    #-----------------------------------------------------------#
    density = 1.1
    box_length = np.sqrt(N / density)
    r_1p1 = init_config(N)
    h = histogram(limits=np.array([0.1, 3.5]), binwidth=0.1)
    h = fill_h(r_1p1, h, box_length)
    for sweeps in range(N_sweeps):
        r_1p1 = MC_sweep(r_1p1, box_length, max_hop_length, T)
        h = fill_h(r_1p1, h, box_length)
    plot_rdf(h, N, N_sweeps, density, 'ps11_problem2vib.png')
    draw_config(r_1p1, box_length, 'Configuration of density=1.1')

    #vii)
    print('''vii) The configuration at density 1.1 resembles a much more ordered structure than the liquid
    density 0.7, as expected. When looking at a tagged particle, the solid structure has a more uniform neighbour 
    set than that of liquids. The rdf of solid also shows a higher magnitude at r near the particle diameter, 
    indicating a more closed packed structure.''')
