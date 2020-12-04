import numpy as np
import matplotlib.pyplot as plt
from numba import jit

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
def compute_forces_and_potential(r, box_length):
    '''calculates the F and U of a Lennard Jones system'''
    N = r.shape[0]
    r_cut = box_length / 2
    r_cut_squared = r_cut ** 2
    forces = np.zeros((N, 2))
    potential = 0
    for i in range(N):
        for j in range(i + 1, N):
            # compute the distance between particle i and j and stores the F and U values. Fij = Fji due to Newton
            # third law
            dr = r[i, :] - r[j, :]
            dr -= box_length * np.floor(dr / box_length + 0.5)
            dr2 = dr @ dr
            dr2 = min(dr2, r_cut_squared)
            # periodic condition
            if dr2 < r_cut_squared:
                force_factor = 48 * (dr2 ** (-7) - 0.5 * dr2 ** (-4))
                forces[i, :] += force_factor * dr
                forces[j, :] += force_factor * (-dr)
                potential += 4 * (dr2 ** (-6) - dr2 ** (-3)) - 4 * (r_cut_squared ** (-6) - r_cut_squared ** (-3))

    return forces, potential

def fill_h(r, h, box_length):
    N = r.shape[0]
    r_cut = box_length / 2
    for i in range(N):
        for j in range(N):
            if j > i:
                dr = r[i, :] - r[j, :]
                dr -= box_length * np.floor(dr / box_length + 0.5)
                dr = np.linalg.norm(dr)
                if dr < r_cut:  # selection rule
                    h.add_sample(dr)
    return h

def do_MD(r, h, delta_t, k_coll, box_length, N, T, N_steps):
    forces, potential_energy = compute_forces_and_potential(r, box_length)
    # initialize K and U trajectory
    v = np.zeros((N, 2))
    for step in range(N_steps):
        # Implementation of verlet algorithm to advance time
        v = v + 0.5 * delta_t * forces
        r = r + delta_t * v
        forces, potential_energy = compute_forces_and_potential(r, box_length)
        v = v + 0.5 * delta_t * forces
        if step % 10 == 0 and step/N_steps > 0.1:
            h = fill_h(r, h, box_length)
        # Implementation of Andersen's thermostat where we resample velocity to account for collision
        for i in range(N):
            if np.random.rand() < k_coll * delta_t:  # draw sample
                speed = np.sqrt(-2 * T * np.log(np.random.rand()))  # new speed
                angle = 2 * np.pi * np.random.rand()  # new angle
                v[i, :] = speed * np.array([np.cos(angle), np.sin(angle)])  # new velocity by 'vectorizing' speed with new
                # angle

    return h

def plot_rdf(h, N, N_steps, density, title):
    '''Plot the RDF from histogram H'''
    g = h.histo / (N_steps * np.pi * h.vals * h.binwidth * (N - 1) * density)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, title='rdf g(r) at density {0}'.format(density), ylabel='g(r)', xlabel='r')
    ax.bar(h.vals, g, width=0.95 * h.binwidth, color='k')
    plt.savefig(title)
    plt.show()


if __name__ == '__main__':
    # trial calculation
    #N = 16
    #density = 0.7
    #box_length = np.sqrt(N / density)
    #r = init_config(N)
    #print(compute_forces_and_potential(r, box_length, N))
    print('''comment see LJ_cluster_MD.py''')
    #iii)
    N = 36
    delta_t = 0.01
    total_t = 1000
    N_steps = int(total_t // delta_t)
    density = 0.7
    k_coll = 1
    T = 0.7
    box_length = np.sqrt(N / density)
    h = histogram(limits=np.array([0, 3.5]), binwidth=0.1)
    r = init_config(N)
    h = do_MD(r, h, delta_t, k_coll, box_length, N, T, N_steps)
    plot_rdf(h, N, N_steps, density, 'ps12_problem1iii.png')
    print('''iii) Compared to previous assignment's MC simulation, the MD result yields a very similar plot of the 
    radial distribution function. ''')
