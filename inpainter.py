import numpy as np
from matplotlib import pyplot as plt
import sys
import math
import cv2 as cv

from patch_extractor import patch_extractor
from scale import reduce_matrix, expand_matrix
from skimage.restoration import inpaint

DEBUG = False
DEBUG_K = False


class Inpainter:
    def __init__(self, matrix, target_region_coords):
        self._matrix = np.array(matrix)
        self._target_region_coords = np.array(target_region_coords)

    def restore(self):
        scaled_mats = [self._matrix.copy()]
        tr_coords = [self._target_region_coords.copy()]

        for i in range(2):
            # obtain channels matrices
            channels = scaled_mats[i][:, :, 0], scaled_mats[i][:, :, 1], \
                       scaled_mats[i][:, :, 2]
            # resize each channel matrix
            reduced_channels = [reduce_matrix(channel) for channel in channels]
            # join channel mats
            reduced_mat = np.dstack(reduced_channels)
            scaled_mats.append(reduced_mat)
            tr_coords.append(tr_coords[i] // 2)

        # apply harmonic inpainting on l2
        harmonic_mask = np.zeros(scaled_mats[2].shape)
        harmonic_mask[tr_coords[2][0]:tr_coords[2][2],
        tr_coords[2][1]:tr_coords[2][3]] = 1

        harmonic_mat = inpaint.inpaint_biharmonic(self._matrix, harmonic_mask)
        harmonic_mat = harmonic_mat * 255
        harmonic_mat = harmonic_mat.astype(np.uint8)

        # inpaint l2 = approximation matrix + patches
        cur_mat = self.inpaint(harmonic_mat, tr_coords[2])
        for i in [1, 0]:
            # make x2
            adv_mat = expand_matrix(cur_mat)
            # extract w + insert in next l
            adv_tr = adv_mat[tr_coords[i][0]:tr_coords[i][2],
                     tr_coords[i][1]:tr_coords[i][3]]
            cur_mat = scaled_mats[1]
            cur_mat[tr_coords[i][0]:tr_coords[i][2],
            tr_coords[i][1]:tr_coords[i][3]] = adv_tr
            # inpaint next l
            cur_mat = self.inpaint(cur_mat, tr_coords[i])

        return cur_mat

    def inpaint(self, matrix, target_region_coords):
        matrix = matrix.copy()
        p_size = 10
        # convert to grayscale
        gray_matrix = cv.cvtColor(matrix, cv.COLOR_BGR2GRAY)
        # find appr matrix
        appr_mat = self.calculate_approximation_matrix(gray_matrix,
                                                       target_region_coords)
        S_ps = []
        mat_p_i, mat_p_j = target_region_coords[0], target_region_coords[1]
        while mat_p_i < target_region_coords[2]:
            while mat_p_j < target_region_coords[3]:
                # extract pes
                W_p = appr_mat[mat_p_i:mat_p_i + p_size, mat_p_j:mat_p_j + p_size]
                # compare with all S patches
                compared = [(S_p, self.calculate_ssim([S_p], [W_p])) for S_p in S_ps]
                compared = sorted(compared, key=lambda x: x[1], reverse=True)
                # select first + prop values
                matrix[mat_p_i:mat_p_i + p_size, mat_p_j:mat_p_j + p_size] \
                    = # patch from color image at coords of compared[0][0]
                mat_p_j += p_size
            mat_p_i += p_size
            mat_p_j = target_region_coords[1]
        return matrix

    def calculate_approximation_matrix(self, matrix, target_region_coords):
        target_region_coords = ((target_region_coords[0], target_region_coords[1]),
                                (target_region_coords[2], target_region_coords[3]))

        U, s, VT = np.linalg.svd(matrix, full_matrices=False)

        Sigma = np.zeros((matrix.shape[0], matrix.shape[1]))
        Sigma[:min(matrix.shape[0], matrix.shape[1]),
        :min(matrix.shape[0], matrix.shape[1])] = np.diag(s)

        k = s.shape[0]
        ssim = 1
        prev_mat = matrix
        cur_mat = matrix
        init_patches = patch_extractor.get_patches(matrix, target_region_coords)

        while True:
            if ssim < 0.3:
                break
            prev_mat = cur_mat
            cur_mat = U[:, :k - 1] @ Sigma[:k - 1, :k - 1] @ VT[:k - 1, :]
            cur_mat = np.round(cur_mat)
            cur_mat = cur_mat.astype(np.uint8)

            cur_patches = patch_extractor.get_patches(cur_mat, target_region_coords)
            ssim = self.calculate_ssim(init_patches, cur_patches)
            k -= 1

            if DEBUG:
                if k % 1 == 0:
                    cv.imshow(f'SV: {k - 1}, ssim : {ssim}', cur_mat)
                    cv.waitKey(0)
                pass

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
