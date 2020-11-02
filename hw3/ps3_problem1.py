import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def get_z(x, y):
    """Returns f(x, y) = sin(xy) * exp(-\sqrt{x^2 + y^2})
    Parameters
    ----------
    x  array
    y  array

    Returns
    -------
    array
    """
    return np.sin(x * y) * np.exp(-np.sqrt(x**2 + y**2))

def get_z1(x, y):
    """Returns f(x, y) = \sum_k\in odd ^ 9
    \sum_m\in odd ^ 9 \frac{sin(kx)sin(my)}{km}
    Parameters
    ----------
    x  array
    y  array

    Returns
    -------
    array
    """
    odd = np.arange(1, 10)
    odd = odd[odd % 2 == 1]
    result = 0
    for k in odd:
        for m in odd:
            result += np.sin(k*x) * np.sin(m*y) / (k*m)
    return result

if __name__ == '__main__':
    #ii)
    xq = np.linspace(-4, 4)
    yq = np.linspace(-4, 4)
    xx, yy = np.meshgrid(xq, yq)
    z = get_z(xx, yy)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, z, cmap=cm.plasma)
    plt.title('Surface plot of $f(x, y) = sin(xy)exp(-\sqrt{x^2 + y^2})$')
    plt.savefig('ps3_problem1_ii.png')
    plt.show()

    #iii)
    xq = np.linspace(-4, 4, 500)
    yq = np.linspace(-4, 4, 500)
    xx, yy = np.meshgrid(xq, yq)
    z = get_z(xx, yy)
    fig1 = plt.figure()
    plt.contourf(xx, yy, z, levels=20, cmap='seismic')
    plt.title('Contour plot of $f(x, y) = sin(xy)exp(-\sqrt{x^2 + y^2})$')
    plt.axis('equal')
    plt.savefig('ps3_problem1_iii.png')
    plt.show()

    #iv)
    xq = np.linspace(-np.pi, np.pi, 500)
    yq = np.linspace(-np.pi, np.pi, 500)
    xx, yy = np.meshgrid(xq, yq)
    z = get_z1(xx, yy)
    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    ax2.plot_surface(xx, yy, z, cmap=cm.plasma)
    plt.title('Surface plot of $f(x, y) = \sum_{k\in odd}^9 \
    \sum_{m\in odd}^9 \\frac{sin(kx)sin(my)}{km}$')
    plt.savefig('ps3_problem1_ivsurf.png')
    plt.show()
    fig3 = plt.figure()
    plt.contourf(xx, yy, z, levels=20, cmap='seismic')
    plt.axis('equal')
    plt.title('Contour plot for $f(x, y) = \sum_{k\in odd}^9 \
    \sum_{m\in odd}^9 \\frac{sin(kx)sin(my)}{km}$')
    plt.savefig('ps3_problem1_ivcont.png')
    plt.show()