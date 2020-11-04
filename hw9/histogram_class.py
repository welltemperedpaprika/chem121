# import packages for basic math, plotting, linear algebra, etc.
from numpy import *
from numpy.linalg import *
from numpy.random import *
from matplotlib.pyplot import *
from scipy.special import binom, erf, erfc

class histogram():
    '''Histogram class.'''
    def __init__(self, limits, binwidth):
        '''Initializes the histogram object (myself).

        Parameters
        ----------
        limits, set the limits of histogram
        binwidth, set the bin width of histogram
        ----------
        '''
        self.limits = limits
        self.binwidth = binwidth
        self.vals = arange(self.limits[0] + self.binwidth / 2, self.limits[1], self.binwidth)
        self.histo = 0 * self.vals
        self.N_samples = 0

    def add_sample(self,dat):
        '''Adds a sample to myself if the sample falls within my limits.

        Parameters
        ----------
        dat, the sample to be added
        '''
        self.N_samples += 1
        if dat > self.limits[0] and dat < self.limits[1]:
            bin_index = int((dat - self.limits[0]) / self.binwidth)
            self.histo[bin_index] += 1

    def normalize(self):
        '''Normalizes myself such that the area under me is 1.'''
        self.histo = self.histo / (self.N_samples * self.binwidth)

    def barplot(self):
        '''Plots a barplot from myself.'''
        bar(self.vals, self.histo, width=0.95 * self.binwidth, color='k')

    def compute_mean(self):
        '''Computes my mean.'''
        self.mean = sum(multiply(self.vals, self.histo) * self.binwidth)
        return self.mean

    def compute_mean_square(self):
        '''Computes my mean squared, \langle X^2 \rangle'''
        self.mean_square = sum(multiply(self.vals ** 2, self.histo) * self.binwidth)
        return self.mean_square

    def compute_std_dev(self):
        '''Computes my standard deviation'''
        self.std_dev = sqrt(self.mean_square - self.mean ** 2)
        return self.std_dev

    def compute_error_of_mean(self):
        '''Computes my error of mean'''
        self.error_of_mean = self.std_dev / sqrt(self.N_samples)
        return self.error_of_mean

    def compute_histogram_error(self):
        '''Computes my error at each bin of histogram'''
        histogram_err = 0 * self.histo
        for c, x in enumerate(self.histo):
            histogram_err[c] = sqrt(x*(1 - x * self.binwidth) / (self.N_samples * self.binwidth))
        return histogram_err

    def plot_error_bars(self):
        '''Makes a plot using the 2*errors of each bin as error bars.'''
        y_err = 2 * self.compute_histogram_error()
        errorbar(self.vals, self.histo, y_err, fmt='.')
        xlabel('x')
        ylabel('normalized count')


def example():
    # makes a histogram object.
    h = histogram(limits=[0,1],binwidth=0.1)

    N_samples = 100000
    for sample in range(N_samples):
        # draw N_samples from a uniform dist. to add to histogram h.
        dat = rand()
        h.add_sample(dat)

    # plot the histogram object as a barplot.
    h.normalize()
    print(h.compute_mean())
    clf()
    h.barplot()
    print(h.histo)
