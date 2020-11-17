# import packages for basic math, plotting, linear algebra, etc.
from numpy import *
from numpy.linalg import *
from numpy.random import *
from matplotlib.pyplot import *
from scipy.special import binom, erf, erfc
from numba import jit

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

    def lineplot(self):
        plot(self.vals, self.histo)


@jit(nopython=True)
def MC_sweep(n_quanta):
    for step in range(M):
        donor = int(M * rand())
        acceptor = int(M * rand())

        if n_quanta[donor] > 0:
            n_quanta[donor] -= 1
            n_quanta[acceptor] += 1

    return n_quanta

if __name__ == '__main__':

    energy_per_particle = 3 # E/M, assume this is an integer

    M = 100
    E = M * energy_per_particle

    h = histogram(limits=[-0.5,E+0.5],binwidth=1)

    n_quanta = ones(M).astype(int) * energy_per_particle

    N_sweeps = 100000
    for sweep in range(N_sweeps):
        # for step in range(M):
        #     donor = int(M * rand())
        #     acceptor = int(M * rand())
        #
        #     if n_quanta[donor] > 0:
        #         n_quanta[donor] -= 1
        #         n_quanta[acceptor] += 1
        n_quanta = MC_sweep(n_quanta)
        h.add_sample(n_quanta[0])

    # clf()
    h.normalize()
    h.lineplot()
    xlim(0,20)
    xlabel(r'$n_1$',fontsize=14)
    ylabel(r'p($n_1)$',fontsize=14)

    beta = log(1 + M/E)
    Boltz_dist = exp(-beta * h.vals) * (1 - exp(-beta))
    plot(h.vals,Boltz_dist,'o-')