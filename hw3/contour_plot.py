# import packages for basic math, plotting, etc.
from numpy import *
from matplotlib.pyplot import *

#  set x and y values to query
x_values = arange(-2,2,0.1)
y_values = x_values
number_of_values = size(x_values)

#  set an n x n matrix where each entry corresponds to the z value of (x, y)
example_function = zeros((number_of_values,number_of_values))

#  allocate z values to matrix
for i in range(number_of_values):
    x = x_values[i]
    for j in range(number_of_values):
        y = y_values[j]

        example_function[i,j] = x**2 + y**2

clf()
#  plots contour given x and y and z values
contourf(x_values, y_values, example_function, levels=20, cmap=cm.seismic)
axis('equal')
show()