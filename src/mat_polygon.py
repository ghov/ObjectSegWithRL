import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import math

fig, ax = plt.subplots()
patches = []
num_polygons = 1
num_sides = 4

size = (8.0, 4.0*(math.sqrt(5)-1))

fig = plt.figure(1, figsize=size, dpi=90)

# for i in range(num_polygons):
#     polygon = Polygon(np.random.rand(num_sides ,2), True)
#     patches.append(polygon)

custom_arr = np.array([[0, 0], [0, 2], [2, 2], [1, 0]])

xrange = [-1, 3]
yrange = [-1, 3]
ax.set_xlim(*xrange)
ax.set_ylim(*yrange)
ax.set_aspect(1)



print(custom_arr)
print(custom_arr.shape)
#rand_arr = np.random.rand(num_sides, 2)

#print(rand_arr)
polygon = Polygon(custom_arr, True)
patches.append(polygon)

print(polygon.contains_point())
polygon.contains_point([.5, .5])
p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)

colors = 100*np.random.rand(len(patches))
p.set_array(np.array(colors))

ax.add_collection(p)
#plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.show()