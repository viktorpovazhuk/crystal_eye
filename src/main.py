import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
from inpainter import Inpainter

filename = 'bird'

mat = cv.imread('data/' + filename + '.jpg', cv.IMREAD_COLOR)
# (top-right row-col, bottom-left row-col)
# (i1, j1, i2, j2)
# cover, butter
# W_coords = np.array((100, 130, 160, 190))
# # boy
# W_coords = np.array((180, 28, 205, 50))
# W_coords = np.array((105, 82, 120, 97))
# # bee
# W_coords = np.array((760, 1100, 860, 1325))
# bird
W_coords = np.array((65, 145, 80, 200))
mat[W_coords[0]:W_coords[2], W_coords[1]:W_coords[3]] = 255
patch_size = 6
x_step, y_step = 4, 4
beta = 0.9

inp = Inpainter()
inp.set_params(beta)

restored_mat = inp.restore(mat, W_coords, patch_size, x_step, y_step)
# restored_mat = inp.inpaint(harmonic_mat, W_coords, patch_size, x_step, y_step)

cur_time = time.strftime("%H_%M_%S")

plt.imshow(restored_mat[..., ::-1])
plt.savefig(f'results/{filename}/{filename}_{cur_time}.png')
plt.show()

with open(f'results/{filename}/params.txt', 'a') as f:
    f.write(f'{filename}_{cur_time}:\n'
            f'patch_size: {patch_size}\n'
            f'x_step: {x_step}, y_step: {y_step}\n'
            f'beta: {beta}\n'
            f'--------------------\n\n')
