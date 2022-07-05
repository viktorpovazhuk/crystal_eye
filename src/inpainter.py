import numpy as np
from matplotlib import pyplot as plt
import sys
import math
import cv2 as cv
import time

from harmonic_inpainting import harmonic
from patch_extractor import PatchExtractor
from scale import reduce_matrix, expand_matrix, expand_color_matrix, reduce_color_matrix
from skimage.restoration import inpaint

DEBUG = True
path_prefix = 'results/test/'


class Inpainter:
    def __init__(self):
        self._beta = 0.8

    def set_params(self, beta):
        self._beta = beta

    def restore(self, matrix, W_coords, patch_size, x_step, y_step):
        # 3 levels of matrices and parameters
        scaled_mats = [matrix.copy()]
        scaled_W_coords = [W_coords.copy()]
        patch_sizes = [patch_size]
        x_steps, y_steps = [x_step], [y_step]

        for i in range(2):
            scaled_mats.append(reduce_color_matrix(scaled_mats[i]))
            cur_W_coords = scaled_W_coords[i] // 2
            cur_W_coords[-2:] += 1
            cur_W_coords[:2] -= 1
            scaled_W_coords.append(cur_W_coords)
            patch_sizes.append(max(patch_sizes[i] // 2, 3))
            x_steps.append(max(x_steps[i] // 2, 2))
            y_steps.append(max(y_steps[i] // 2, 2))

        # apply harmonic inpainting on l2
        harmonic_mat = harmonic(scaled_mats[2], scaled_W_coords[2])

        if DEBUG:
            plt.title('harmonic')
            plt.imshow(harmonic_mat)
            plt.savefig(f'{path_prefix}/{time.strftime("%H_%M_%S")}.png')
            plt.show()

        # inpaint l2 = approximation matrix + patches
        cur_mat = self.inpaint(harmonic_mat, scaled_W_coords[2], patch_sizes[2], x_steps[2], y_steps[2])

        if DEBUG:
            plt.title('inpainted L2')
            plt.imshow(cur_mat)
            plt.savefig(f'{path_prefix}/{time.strftime("%H_%M_%S")}.png')
            plt.show()

        for i in [1, 0]:
            # make x2
            adv_mat = expand_color_matrix(cur_mat)

            # extract w + insert in next l
            adv_W = adv_mat[scaled_W_coords[i][0]:scaled_W_coords[i][2],
                     scaled_W_coords[i][1]:scaled_W_coords[i][3]]
            cur_mat = scaled_mats[i]
            cur_mat[scaled_W_coords[i][0]:scaled_W_coords[i][2],
            scaled_W_coords[i][1]:scaled_W_coords[i][3]] = adv_W

            # inpaint next l
            cur_mat = self.inpaint(cur_mat, scaled_W_coords[i], patch_sizes[i], x_steps[i], y_steps[i])

            if DEBUG:
                plt.title(f'inpainted L{i}')
                plt.imshow(cur_mat)
                plt.savefig(f'{path_prefix}/{time.strftime("%H_%M_%S")}.png')
                plt.show()

        return cur_mat

    def inpaint(self, matrix, W_coords, patch_size, x_step, y_step):
        matrix = matrix.copy()

        # convert to grayscale
        gray_matrix = cv.cvtColor(matrix, cv.COLOR_BGR2GRAY)
        # find appr matrix
        appr_mat = self.calculate_approximation_matrix(gray_matrix,
                                                       W_coords)

        if DEBUG:
            plt.title('approximation')
            plt.imshow(appr_mat, cmap='gray')
            plt.savefig(f'{path_prefix}/{time.strftime("%H_%M_%S")}.png')
            plt.show()

        # extract patches from S
        patch_extractor = PatchExtractor()
        S_ps = patch_extractor.get_patches_by_size(appr_mat, x_step, y_step, patch_size,
                                                   W_coords)

        # compare and propagate
        matrix = self.populate_patches(matrix, appr_mat, W_coords, S_ps, patch_size)

        return matrix

    def populate_patches(self, matrix, appr_matrix, W_coords, S_patches, patch_size):
        matrix = matrix.copy()

        mat_p_i, mat_p_j = W_coords[0], W_coords[1]
        num_work_patches = ((W_coords[2] - W_coords[0]) // patch_size) * \
                           ((W_coords[3] - W_coords[1]) // patch_size)

        if DEBUG:
            print('patches extracted')
            print(f'number in W: {num_work_patches}')

        p_num = 0
        while mat_p_i < W_coords[2]:
            while mat_p_j < W_coords[3]:
                # extract p
                W_p = appr_matrix[mat_p_i:mat_p_i + patch_size,
                      mat_p_j:mat_p_j + patch_size]
                # compare with all S patches
                compared = [(S_p, self.calculate_ssim([S_p.patch], [W_p])) for S_p in
                            S_patches]
                compared = sorted(compared, key=lambda x: x[1], reverse=True)
                # select first + prop values
                p_coords = compared[0][0].coordinate
                matrix[mat_p_i:mat_p_i + patch_size, mat_p_j:mat_p_j + patch_size] \
                    = matrix[p_coords[0]:p_coords[0] + patch_size,
                      p_coords[1]:p_coords[1] + patch_size]
                mat_p_j += patch_size

                if DEBUG:
                    print(f'{p_num} patch done')
                    p_num += 1
                    if p_num % (num_work_patches // 3) == 0:
                        plt.title('progress')
                        plt.imshow(matrix)
                        plt.show()

            mat_p_i += patch_size
            mat_p_j = W_coords[1]

        return matrix

    def calculate_approximation_matrix(self, matrix, target_region_coords):
        U, s, VT = np.linalg.svd(matrix, full_matrices=False)

        Sigma = np.zeros((matrix.shape[0], matrix.shape[1]))
        Sigma[:min(matrix.shape[0], matrix.shape[1]),
        :min(matrix.shape[0], matrix.shape[1])] = np.diag(s)

        k = s.shape[0]
        ssim = 1
        prev_mat = matrix
        cur_mat = matrix
        init_patches = PatchExtractor.get_patches(matrix, target_region_coords)

        while True:
            if ssim < self._beta:
                break
            prev_mat = cur_mat
            cur_mat = U[:, :k - 1] @ Sigma[:k - 1, :k - 1] @ VT[:k - 1, :]
            cur_mat = np.round(cur_mat)
            cur_mat = cur_mat.astype(np.uint8)

            cur_patches = PatchExtractor.get_patches(cur_mat, target_region_coords)
            ssim = self.calculate_ssim(init_patches, cur_patches)
            k -= 1

        return prev_mat

    def calculate_ssim(self, initial_patches, current_patches):
        c1, c2 = sys.float_info.min, sys.float_info.min
        n = len(initial_patches)
        ssim = 0

        i = 0
        while i < n:
            x = initial_patches[i].flatten()
            y = current_patches[i].flatten()

            if x.shape[0] == 0 or y.shape[0] == 0:
                i += 1
                continue

            cur_ssim = (((2 * np.mean(x) * np.mean(y) + c1) * (
                    2 * np.cov(x, y)[0][1] + c2)) /
                        ((np.mean(x) ** 2 + np.mean(y) ** 2 + c1) * (
                                np.std(x) ** 2 + np.std(y) ** 2 + c2)))

            ssim += cur_ssim
            i += 1
        ssim = ssim / n
        return ssim
