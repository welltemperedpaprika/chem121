import numpy as np
from numpy.linalg import eig

#i)
A = np.array([[3, -0.5, 0, 0, 0],
              [-1/2, 3/2, -1/2, 0, 0],
              [0, -1/2, 1, -1/2, 0],
              [0, 0, -1/2, 3/2, -1/2],
              [0, 0, 0, -1/2, 3]])

#ii)
eval, evec = eig(A)

#iii)
max_i = np.argmax(eval)
E_max = eval[max_i]
c_max = evec[:, max_i]
print('cmax =', c_max,'has eigenvalue', E_max)

#iv)
min_i = np.argmin(eval)
E_min = eval[min_i]
c_min = evec[:, min_i]
print('cmin =', c_min,'has eigenvalue', E_min)

#v)
print('Ac_max =', A @ c_max, 'E_max c_max =', E_max * c_max)
print('Ac_min =', A @ c_min, 'E_min c_min =', E_min * c_min)

#vi)
print('||c_max|| =', c_max.T @ c_max)
print('||c_min|| =', c_min.T @ c_min)

#vii)
print('<c_max | c_min> = ', c_max.T @ c_min)

