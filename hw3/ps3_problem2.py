import numpy as np


def make_M(n, deltax=1, print_=True):
    """Makes matrix M that has first column of [{-n}_{2n+1}, {-n+1}_{2n+1}]...
    and second column of [{-n, -n+1, ..., n}_{2n+1}]

    Parameters
    ----------
    n int
    deltax float
    print_ boolean

    Returns
    -------
    2d array
    """
    query = np.arange(-n*deltax, n*deltax + deltax, deltax)
    xx, yy = np.meshgrid(query, query)
    c1 = xx.flatten('F')
    c2 = yy.flatten('F')
    M = np.column_stack((c1, c2))
    if print_:
        print(M)
    return M

if __name__ == '__main__':
    #i)
    query = np.arange(-1, 2)
    xx, yy = np.meshgrid(query, query)
    c1 = xx.flatten('F')
    c2 = yy.flatten('F')
    M = np.column_stack((c1, c2))
    print("i):", M)

    #ii)
    print("ii): See make_M at line 4 for generalization. e.g. For n=2:")
    make_M(2)
