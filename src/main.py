import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
from inpainter import Inpainter


def main(filename, W_coords, beta, x_step, y_step, patch_size):
    res_prefix = "../results"
    data_prefix = "../data"

    mat = cv.imread(f"{data_prefix}/{filename}", cv.IMREAD_COLOR)

    # (top-right row-col, bottom-left row-col)
    # (i1, j1, i2, j2)
    W_coords = np.array(W_coords)
    mat[W_coords[0]:W_coords[2], W_coords[1]:W_coords[3]] = 255

    inp = Inpainter()
    inp.set_params(beta)

    restored_mat = inp.restore(mat, W_coords, patch_size, x_step, y_step)

    cur_time = time.strftime("%H_%M_%S")

    plt.imshow(restored_mat[..., ::-1])
    plt.savefig(f'{res_prefix}/{filename}/{filename}_{cur_time}.png')
    plt.show()

    with open(f'{res_prefix}/{filename}/params.txt', 'a') as f:
        f.write(f'{filename}_{cur_time}:\n'
                f'patch_size: {patch_size}\n'
                f'x_step: {x_step}, y_step: {y_step}\n'
                f'beta: {beta}\n'
                f'--------------------\n\n')


if __name__ == "__main__":
    main("cover.jpg", (100, 130, 160, 190), 0.8, 4, 4, 6)
