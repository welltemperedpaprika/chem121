# import packages for basic math, plotting, linear algebra, etc.
from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D

# initialize x and y points to query
x_values = arange(-2,2,0.1)
y_values = x_values
number_of_values = size(x_values)

#  set an n x n matrix where each entry corresponds to the z value of (x, y)
example_function = zeros((number_of_values,number_of_values))
#  allocate values
for i in range(number_of_values):
    x = x_values[i]
    for j in range(number_of_values):
        y = y_values[j]

        example_function[i,j] = x**2 + y**2

clf()
#  makes a grid of x and y in the form of repeating numbers down a column for xgrid
#and across a row for ygrid
xgrid, ygrid = meshgrid(x_values, y_values)
ax = axes(projection='3d')
ax.plot_surface(xgrid, ygrid, example_function, cmap=cm.seismic)  #plots the 3d plot
show()