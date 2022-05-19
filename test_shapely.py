from shapely.geometry import Polygon,LineString
from shapely import ops
import matplotlib.pyplot as plt

import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection


# Plots a Polygon to pyplot `ax`
def plot_polygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)
    
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


a = [0,0]
b = [0,1]
c = [1,1]
d = [1,0]

# p = Polygon([a,b,c,d,a])
line = LineString([[0.2,0.2],[0.2,0.4]])
# fig,ax = plt.subplots()

# plot_polygon(ax, p, facecolor='lightblue', edgecolor='red')

# # # ax.plot()
# plt.plot(*line.xy,label="lane")
# plt.legend()
# plt.show()

# print(line.intersects(p))

from shapely import geometry, ops

# create three lines
line_a = geometry.LineString([[0,0], [1,1]])
line_b = geometry.LineString([[1,1], [1,0]])
line_c = geometry.LineString([[1,0], [2,0]])

# combine them into a multi-linestring
multi_line = geometry.MultiLineString([line_a, line_b, line_c])
print(multi_line)  # prints MULTILINESTRING ((0 0, 1 1), (1 1, 2 2), (2 2, 3 3))

# you can now merge the lines
merged_line = ops.linemerge(multi_line)
print(merged_line)  # prints LINESTRING (0 0, 1 1, 2 2, 3 3)

# plt.plot(*merged_line.xy)
# plt.plot(*line.xy)
# intersection = line.intersection(merged_line).boundary

# print(intersection)
# # print(len(intersection))
# print(intersection[0].x, intersection[0].y)

l = [line,line_a]
for e in l:
    print(e)
    x,y = e.boundary
    print(x.distance(y))