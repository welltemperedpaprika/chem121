 # import packages for basic math, plotting, linear algebra, etc.
from numpy import *
from numpy.linalg import *
from numpy.random import *
from matplotlib.pyplot import *
from scipy.special import binom, erf, erfc

class histogram():
    def __init__(self,limits,binwidth):
        self.limits = limits
        self.binwidth = binwidth
        self.vals = arange(self.limits[0] + self.binwidth / 2, self.limits[1], self.binwidth)
        self.histo = 0 * self.vals
        self.N_samples = 0

    def add_sample(self,dat):
        self.N_samples += 1
        if dat > self.limits[0] and dat < self.limits[1]:
            bin_index = int((dat - self.limits[0]) / self.binwidth)
            self.histo[bin_index] += 1

    def normalize(self):
        self.histo = self.histo / (self.N_samples * self.binwidth)

    def barplot(self):
        bar(self.vals, self.histo, width=0.95 * self.binwidth, color='k')

if __name__ == '__main__':
    M = 3 # set number of particles
    E = 2 # set energy of the system

    h = histogram(limits=[-0.5,E+0.5],binwidth=1)

    n_quanta = zeros(M).astype(int) # energies of the particles
    n_quanta[0] = E # all energies at particle 1 initially

    N_steps = 1000 # number of steps
    n_trajectory = zeros(N_steps) # trajectory array
    for step in range(N_steps):
        # random donor and acceptor
        donor = int(M * rand())
        acceptor = int(M * rand())

        if n_quanta[donor] > 0:
            n_quanta[donor] -= 1 # remove energy from donor
            n_quanta[acceptor] += 1 # increase energy for acceptor

        n_trajectory[step] = n_quanta[0] # store the energy change of particle 1
        h.add_sample(n_quanta[0])

    # begin plotting
    clf()
    plot(arange(N_steps), n_trajectory)
    h.normalize()
    h.barplot()