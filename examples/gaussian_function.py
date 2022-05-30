import cv2 as cv
import matplotlib.pyplot as plt
from src import scale

mat = cv.imread('../data/butter.jpg', 0)
print(mat.shape)


def plt_reduced():
    reduced1 = scale.reduce_matrix(mat, 5)
    reduced2 = scale.reduce_matrix(reduced1, 5)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(wspace=0.3, hspace=0.2)

    axs[0].imshow(mat, cmap='gray')
    axs[0].set_title(f'initial')

    axs[1].imshow(reduced1, cmap='gray')
    axs[1].set_title(f'reduce x2')

    axs[2].imshow(reduced2, cmap='gray')
    axs[2].set_title(f'reduce x4')

    plt.show()


def plt_expand():
    expand1 = scale.expand_matrix(mat, 5)
    expand2 = scale.expand_matrix(expand1, 5)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(wspace=0.3, hspace=0.2)

    axs[0].imshow(mat, cmap='gray')
    axs[0].set_title(f'initial')

    axs[1].imshow(expand1, cmap='gray')
    axs[1].set_title(f'expand x2')

    axs[2].imshow(expand2, cmap='gray')
    axs[2].set_title(f'expand x4')

    plt.show()


plt_expand()
