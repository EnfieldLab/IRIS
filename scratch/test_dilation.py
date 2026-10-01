import numpy as np
from scipy.ndimage import distance_transform_edt

mask = np.zeros((10, 10), dtype=int)
mask[2, 2] = 1
mask[7, 7] = 2

bg = (mask == 0)
dists, indices = distance_transform_edt(bg, return_indices=True)

radius = 2
expand = (dists <= radius) & (dists > 0)
new_cyto = mask.copy()

nearest_y, nearest_x = indices
nearest_labels = mask[nearest_y[expand], nearest_x[expand]]
new_cyto[expand] = nearest_labels

print(new_cyto)
