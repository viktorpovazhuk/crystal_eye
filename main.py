import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from inpainter import Inpainter

mat = cv.imread('data/butter.jpg', 0)
print(mat.shape)

mat[90:150, 90:150] = 255

inp = Inpainter(mat, ((90, 90), (150, 150)))
app_mat = inp.calculate_approximation_matrix()

cv.imshow('butterfly', app_mat)
cv.waitKey(delay=2000)
