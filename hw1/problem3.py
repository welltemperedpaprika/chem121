import numpy as np
import matplotlib.pyplot as plt

#i)

t = np.linspace(0, 12*np.pi, 1000)
x = np.sin(t) * (np.exp(2*np.cos(t/2)) - 2*np.cos(8*t) - (np.sin(t/6))**6)
y = np.cos(t) * (np.exp(2*np.cos(t/2)) - 2*np.cos(8*t) - (np.sin(t/6))**6)
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(x, y)
plt.show()


#ii)

c1 = plt.Circle((-2, 6), 0.3, color='k')
c2 = plt.Circle((2, 6), 0.3, color='k')
ax.add_patch(c1)
ax.add_patch(c2)
xq = np.linspace(-3, 3)
yq = 1 + 0.1*xq**2
plt.plot(xq, yq, 'r-', linewidth=5)
plt.show()


#iii)
plt.title('Frog')
plt.savefig('problem3.png')
plt.show()
