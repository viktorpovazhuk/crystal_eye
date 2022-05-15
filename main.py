import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from inpainter import Inpainter
from skimage.restoration import inpaint

mat = cv.imread('data/butter.jpg', 0)
print(mat.shape)

mat[100:160, 130:190] = 255

# cv.imshow('initial', mat)
# cv.waitKey()

mask = np.zeros(mat.shape)
mask[100:160, 130:190] = 1

harm_mat = inpaint.inpaint_biharmonic(mat, mask)
harm_mat = harm_mat * 255
harm_mat = harm_mat.astype(np.uint8)

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

# (top-right row-col, bottom-left row-col)
inp = Inpainter(harm_mat, (100, 130, 160, 190))
app_mat = inp.inpaint()

plt.imshow(app_mat, cmap='gray')
plt.show()
