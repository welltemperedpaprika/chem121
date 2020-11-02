import numpy as np
from numpy import array
from numpy.linalg import norm
from matplotlib import pyplot as plt


def plot_contour(cgf, origin):
    """Plots the contour function of a basis function

    Parameters
    ----------
    cgf  dict, Information of a contracted gaussian function
    origin  array, coordinate of the basis function

    Returns
    -------
    none
    """
    xq = np.linspace(-3, 3)
    yq = xq
    xx, yy = np.meshgrid(xq, yq)
    gf1 = np.exp(-cgf['exponents'][0] * ((xx - cgf['origin'][0][0]) ** 2 + (yy - cgf['origin'][0][1]) ** 2))
    gf2 = np.exp(-cgf['exponents'][1] * ((xx - cgf['origin'][1][0]) ** 2 + (yy - cgf['origin'][1][1]) ** 2))
    phi = cgf['coefficients'][0] * gf1 + cgf['coefficients'][1] * gf2
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='x', ylabel='y',
                    title='Projection of {0} basis function onto xy plane, z=0'.format(cgf['name']))
    nuclei = plt.Circle(origin[0:-1], 0.1, color='r')
    ax.add_patch(nuclei)
    ax.legend([nuclei], ['nuclei'])
    plt.contourf(xx, yy, phi, cmap='GnBu')
    plt.axis('equal')
    plt.savefig('ps6_problem2{0}.png'.format(cgf['name']))
    plt.show()


def get_nrep_energy(molecule):
    """Returns the nuclear repulsion energy.

    Parameters
    ----------
    molecule  dict, a dictionary of nucleis of a molecule with nuclei coord and Z values.

    Returns
    -------
    e  float, the nuclei nuclei repulsion energy
    """
    e = 0.
    nucleis = list(molecule.keys())
    n_nuclei = len(nucleis)
    for i in range(n_nuclei):
        for j in range(n_nuclei):
            if j > i:
                e += molecule[nucleis[i]]['Z'] * molecule[nucleis[j]]['Z'] / \
                     (norm(molecule[nucleis[i]]['origin'] - molecule[nucleis[j]]['origin']))
    return e


if __name__ == '__main__':
    # i)
    # initialize relevant information. Each key is a nuclei with its coordinate and atomic number.
    R_nuclei_dict = {'H1': {'origin': array([1.4305226, 1.10737950, 0.0000]), 'Z': 1},
                     'O': {'origin': array([0.0000, 0.0000, 0.0000]), 'Z': 8},
                     'H2': {'origin': array([-1.4305226, 1.10737950, 0.0000]), 'Z': 1}}
    # begin plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='x ($a_0$)', ylabel='y ($a_0$)'
                         , title='Nuclei Position of H2O on the x-y plane')
    H1 = plt.Circle(R_nuclei_dict['H1']['origin'][0:-1], 0.05, color='b')
    H2 = plt.Circle(R_nuclei_dict['H2']['origin'][0:-1], 0.05, color='b')
    O = plt.Circle(R_nuclei_dict['O']['origin'][0:-1], 0.1, color='r')
    ax.add_patch(H1)
    ax.add_patch(H2)
    ax.add_patch(O)
    plt.axis('equal')
    plt.ylim(-0.5, 1)
    plt.xlim(-2, 2)
    plt.legend([H1, H2, O], ['H1', 'H2', 'O'])
    plt.savefig('Nuclei Positions of H2O')
    plt.show()
    print("""i) The nuclei array is""", R_nuclei_dict)

    # ii)
    # basis set information of list of dictionaries.
    # Each dictionary contains info of one CGF, with keys of the corresponding nuclei,
    # "angular momentum" (the n,l,m factor in a cgf), the coordinate of the cgf, the alpha parameters,
    # and the contraction coefficients.
    basis_set = \
        [{'name': 'H1_1s', 'ang_mom': (0, 0, 0), 'origin': [R_nuclei_dict['H1']['origin']] * 2, 'exponents': [
            1.309756377,
            0.2331359749],
          'coefficients': [0.381381, 0.164964]},
         {'name': 'O_1s', 'ang_mom': (0, 0, 0), 'origin': [R_nuclei_dict['O']['origin']] * 2,
          'exponents': [49.98097117, 8.896587673],
          'coefficients': [0.381381, 0.164964]},
         {'name': 'O_2s', 'ang_mom': (0, 0, 0), 'origin': [R_nuclei_dict['O']['origin']] * 2,
          'exponents': [1.945236531, 0.4933633506],
          'coefficients': [0.0241442, 0.168105]},
         {'name': 'O_2px', 'ang_mom': (1, 0, 0),
          'origin': [R_nuclei_dict['O']['origin'] - [0.1, 0, 0], R_nuclei_dict['O']['origin'] + [0.1, 0, 0]],
          'exponents': [0.9, 0.9],
          'coefficients': [-1, 1]},
         {'name': 'O_2py', 'ang_mom': (0, 1, 0),
          'origin': [R_nuclei_dict['O']['origin'] - [0, 0.1, 0], R_nuclei_dict['O']['origin'] + [0, 0.1, 0]],
          'exponents': [0.9, 0.9],
          'coefficients': [-1, 1]},
         {'name': 'O_2pz', 'ang_mom': (0, 0, 1),
          'origin': [R_nuclei_dict['O']['origin'] - [0, 0, 0.1], R_nuclei_dict['O']['origin'] + [0, 0, 0.1]],
          'exponents': [0.9, 0.9],
          'coefficients': [-1, 1]},
         {'name': 'H2_1s', 'ang_mom': (0, 0, 0), 'origin': [R_nuclei_dict['H2']['origin']] * 2,
          'exponents': [1.309756377,
                        0.2331359749],
          'coefficients': [0.381381, 0.164964]}]

    print("""ii) The basis sets are compiled as: {0}, where coefficients are mapped to the contraction coefficients, 
    origin are mapped to the centers, and exponents are mapped to the widths of gaussians""".format(basis_set))

    # iii, iv)
    plot_contour(basis_set[0], R_nuclei_dict['H1']['origin'])
    plot_contour(basis_set[1], R_nuclei_dict['O']['origin'])
    plot_contour(basis_set[2], R_nuclei_dict['O']['origin'])
    plot_contour(basis_set[3], R_nuclei_dict['O']['origin'])
    plot_contour(basis_set[4], R_nuclei_dict['O']['origin'])
    plot_contour(basis_set[5], R_nuclei_dict['O']['origin'])
    plot_contour(basis_set[6], R_nuclei_dict['H2']['origin'])

    # v)
    print("""v) The nuclear repulsion energy is {0} hartrees""".format(get_nrep_energy(R_nuclei_dict)))
