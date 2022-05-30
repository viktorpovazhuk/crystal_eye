import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from harmonic_inpainting import harmonic
from inpainter import Inpainter
from scale import reduce_color_matrix, expand_color_matrix

filename = 'butter'

mat = cv.imread('../data/' + filename + '.jpg', cv.IMREAD_COLOR)
W_coords = np.array((100, 130, 160, 190))
mat[W_coords[0]:W_coords[2], W_coords[1]:W_coords[3]] = 255

mat = reduce_color_matrix(mat)
mat = reduce_color_matrix(mat)

for i in range(2):
    W_coords = W_coords // 2
    W_coords[-2:] += 1
    W_coords[:2] -= 1

# print(W_coords)
# mat[W_coords[0]:W_coords[2], W_coords[1]:W_coords[3]] = 0

mat = harmonic(mat, W_coords)
plt.imshow(mat)
plt.show()