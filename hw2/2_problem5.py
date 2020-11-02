# noinspection PyUnresolvedReferences
from parameters import a, b, c, A, B, C
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 2*np.pi, 500)  #set points to query

if __name__ == '__main__':
    x = [np.sum(a * np.sin(b + c * ti)) for ti in t]  #get x from equation
    y = [np.sum(A * np.sin(B + C * ti)) for ti in t]  #get y from equation
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('An Elephant')
    plt.savefig('2_problem5.png')
    plt.show()