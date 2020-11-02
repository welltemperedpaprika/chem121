import numpy as np
import matplotlib.pyplot as plt

xq = np.linspace(-3, 3, 1000)
f1 = 1/2*xq**2
#i)
plt.plot(xq, f1)


#ii)
plt.hlines((1/2, 3/2, 5/2), -3, 3, 'k')

#iii)
g1 = 1/2 + 1/2*np.exp(-xq**2/2)
g2 = 3/2 + 1/2*xq*np.exp(-xq**2/2)
g3 = 5/2 + 1/2*(2*xq**2 - 1)*np.exp(-xq**2/2)
plt.plot(xq, g1, 'r')
plt.plot(xq, g2, 'r')
plt.plot(xq, g3, 'r')
plt.savefig('problem4.png')
plt.show()