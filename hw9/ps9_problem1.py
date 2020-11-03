import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from histogram_class import histogram

def compute_statistics(n, width=0.1):
    h = histogram(limits=[0, 1], binwidth=width)
    for n in range(int(n)):
        dat = np.random.rand()
        h.add_sample(dat)
    h.normalize()
    mean = h.compute_mean()
    mean_square = h.compute_mean_square()
    std_dev = h.compute_std_dev()
    error_of_mean = h.compute_error_of_mean()
    return mean, mean_square, std_dev, error_of_mean

def make_hist_err_plot(n, width=0.1):
    h = histogram(limits=[0, 1], binwidth=width)
    for n in range(int(n)):
        dat = np.random.rand()
        h.add_sample(dat)
    h.normalize()
    h.barplot()
    h.plot_error_bars()
    plt.hlines(1, h.limits[0], h.limits[1], color='orange', label='true p(X)')
    plt.legend()


if __name__ == '__main__':
    #i)
    print("""See histogram_class.py for question 1i) - iii).""")

    #iv-v)
    '''Nsamp = [100, 10e3, 10e4, 10e5, 10e6]
    means = []
    mean_squares = []
    std_devs = []
    errors_of_mean = []
    for n in Nsamp:
        mean, mean_square, std_dev, error_of_mean = compute_statistics(n)
        means.append(mean)
        mean_squares.append(mean_square)
        std_devs.append(std_dev)
        errors_of_mean.append(error_of_mean)
    table = pd.DataFrame({'Nsamp': Nsamp, 'mean': means, 'mean_squares': mean_squares, 'std_devs': std_devs
                         ,'error_of_mean': errors_of_mean})
    print("""iv-v) The error of mean gets smaller as the mean of the sample gets closer to the theoretical value.
    Shown in the table below:
    {0}""".format(table))'''

    #vi - vii)
    print("""vi-vii) See histogram_class.py for method implemented.""")

    #viii)
    Nsamp = [100, 10e3, 10e4, 10e5, 10e6]
    for c, n in enumerate(Nsamp):
        make_hist_err_plot(n)
        plt.savefig('ps9_problem1viii_{0}.png'.format(c))
        plt.show()
    print("""viii) From the plots, the error bars diminishes as Nsamp get larger,
     which is in line with expectation and gives a good sense of inaccuracies.""")

    #ix)
    widths = [0.01, 0.05, 0.1, 0.25]
    for c, w in enumerate(widths):
        make_hist_err_plot(10e4, w)
        plt.savefig('ps9_problem1ix_{0}.png'.format(c))
        plt.show()
    print("""ix) When using wide-bins, the computational time is significantly better than using narrow bin widths.""")

    #x)
    print("""x) If the distribution is non uniform, a small bin width might not capture the true shape of the
    distribution, even at larger Nsamp. To establish a good width, if we know the true distribution, then we
    can iterate over possible values of widths and compare to the true distribution to choose a good balance between
    computational time and representativeness.""")
