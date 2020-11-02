import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 2*np.pi, 1000)
x = 27/5 * np.sin(3/2 - 30*t)
y = 13/7 * np.sin(1/2 - 40*t)

#i)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')


#ii)
plt.title('a pretzel')
plt.savefig('problem2.png')
plt.show()