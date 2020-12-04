# import packages for basic math, plotting, linear algebra, etc.
from numpy import *
from numpy.linalg import *
from numpy.random import *
from matplotlib.pyplot import *
from scipy.special import binom, erf, erfc

class histogram():
    ## a histogram object
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

def plot_circle(center,radius):
    '''plots a circle at CENTER with RADIUS'''
    npoints = 100
    theta = arange(0,2*pi + 1e-7,2*pi/npoints)
    x = center[0] + radius*cos(theta)
    y = center[1] + radius*sin(theta)
    plot(x,y,'k',linewidth=2)

def init_config():
    '''initialize configuration'''
    r = zeros((N, 2))  # array of x, y positions of particles
    n_side = int(sqrt(N) + 0.99) # start with a square lattice
    count = 0
    # allocate the position arrays
    for row in range(n_side):
        for column in range(n_side):
            if count < N:
                r[count, :] = [row, column]
                count += 1

    return r

def draw_config():
    '''draw the configuration on a graph'''
    clf()
    for i in range(N):
        r[i,:] -= box_length * floor(r[i,:]/box_length + 0.5)
        plot_circle(r[i, :], 0.5)
    axis('equal')
    gca().set_adjustable("box")
    view_scale = 1.1*box_length/2
    xlim(-view_scale, view_scale)
    ylim(-view_scale, view_scale)

    boundary_x = box_length * (array([0,1,1,0,0]) - 0.5)
    boundary_y = box_length * (array([0,0,1,1,0]) - 0.5)
    plot(boundary_x,boundary_y)

    pause(0.01)

N = 64 # set number of particles
T = 0.5 # set low temperature
density = 0.7
box_length = sqrt(N/density)


max_hop_length = 0.1
r_cut = box_length/2
r_cut_squared = r_cut**2

r = init_config()
draw_config()

from numba import jit
@jit(nopython=True)
def MC_sweep(r):
    total_U = 0
    for step in range(N):
        i_trial = int(rand()*N) # set trial random particle
        r_trial = r[i_trial,:] + max_hop_length * (rand(2) - 0.5)  # set random movement to a trial location

        delta_U = 0 # begin calculating change in energy
        for j in range(N):
            if j != i_trial:
                dr_trial = r_trial - r[j,:] # vector distance between ij
                dr_trial -= box_length * floor(dr_trial/box_length + 0.5) # account for periodicity
                dr2_trial = dr_trial @ dr_trial
                dr2_trial = min(dr2_trial,r_cut_squared)

                dr = r[i_trial,:] - r[j,:]
                dr -= box_length * floor(dr/box_length + 0.5)
                dr2 = dr @ dr
                dr2 = min(dr2,r_cut_squared)

                delta_U += 4*( dr2_trial**(-6) - dr2_trial**(-3) ) \
                    - 4*( dr2**(-6) - dr2**(-3) )


        if rand() < exp(-delta_U / T): # selection rule
            r[i_trial,:] = r_trial

    return r

N_sweeps = 10000
for sweep in range(N_sweeps):
    r = MC_sweep(r)

    if sweep % 1000 == 0: # draw every 1000 sweeps
        draw_config()

