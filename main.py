import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from inpainter import Inpainter


mat = cv.imread('data/butter.jpg', cv.IMREAD_UNCHANGED)

# cv.imshow('initial', mat)
# cv.waitKey()

# (top-right row-col, bottom-left row-col)
# (i1, j1, i2, j2)
target_region = (100, 130, 160, 190)

mat[target_region[0]:target_region[2], target_region[1]:target_region[3]] = 255

# cv.imshow('harmonic', harm_mat)
# cv.waitKey()

# --------------------------------------------

# U, D, V = np.linalg.svd(mat)
# m, n = mat.shape
#
# A_reconstructed = U[:,:n] @ np.diag(D) @ V[:m,:]
# A_reconstructed = np.round(A_reconstructed)
# A_reconstructed = A_reconstructed.astype(np.uint8)
#
# print(mat.dtype, A_reconstructed.dtype)

# --------------------------------------------

# A_reconstructed = A_reconstructed.flatten()
# mat = mat.flatten()

# for i in range(len(mat)):
#     if A_reconstructed[i] != mat[i]:
#         print(f'{A_reconstructed[i]} : {mat[i]}')

# cv.imshow('butterfly', A_reconstructed)
# cv.waitKey()
# cv.imshow('butterfly', mat)
# cv.waitKey()

# --------------------------------------------


inp = Inpainter(mat, target_region)
restored_mat = inp.restore()

plt.imshow(restored_mat)
plt.show()
