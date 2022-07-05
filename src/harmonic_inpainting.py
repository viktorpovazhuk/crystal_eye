import numpy as np
from matplotlib import pyplot as plt
from skimage.restoration import inpaint


def harmonic(matrix, W_coords):
    harmonic_mask = np.zeros(matrix.shape[:-1], dtype=bool)
    harmonic_mask[W_coords[0]:W_coords[2],
    W_coords[1]:W_coords[3]] = 1

    harmonic_mat = inpaint.inpaint_biharmonic(matrix, harmonic_mask,
                                              channel_axis=-1)
    harmonic_mat = harmonic_mat * 255
    harmonic_mat = harmonic_mat.astype(np.uint8)

    return harmonic_mat
