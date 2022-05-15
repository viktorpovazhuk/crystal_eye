import numpy as np
from matplotlib import pyplot as plt
import sys
import math
import cv2 as cv

from patchExtractor import patch_extractor
from scale import reduce_matrix, expand_matrix

DEBUG = False
DEBUG_K = False


class Inpainter:
    def __init__(self, matrix, target_region_mask):
        self._matrix = np.array(matrix)
        self._target_region_mask = np.array(target_region_mask)

    def inpaint(self):
        appr_mat = self.calculate_approximation_matrix(self._matrix,
                                                       self._target_region_mask)
        return appr_mat

    def inpaint_with_scaling(self):
        reduced_mats = [self._matrix.copy()]
        tr_masks = [self._target_region_mask.copy()]
        for i in range(2):
            reduced_mat = reduce_matrix(reduced_mats[i])
            reduced_mats.append(reduced_mat)
            tr_masks.append(tr_masks[i] // 2)

        for i in [2, 1]:
            cur_appr_mat = self.calculate_approximation_matrix(reduced_mats[i],
                                                               tr_masks[i])
            # plt.imshow(cur_appr_mat, cmap='gray')
            # plt.show()
            for j in range(tr_masks[i].shape[0]):
                if tr_masks[i - 1][j] % 2 == 1:
                    tr_masks[i][j] += 1
            cur_tr = cur_appr_mat[tr_masks[i][0]:tr_masks[i][2] + 1,
                     tr_masks[i][1]:tr_masks[i][3] + 1]
            next_tr = expand_matrix(cur_tr)
            next_tr = next_tr[0:tr_masks[i - 1][2] - tr_masks[i - 1][0] + 1,
                      0:tr_masks[i - 1][3] - tr_masks[i - 1][1] + 1]
            reduced_mats[i - 1][tr_masks[i - 1][0]:tr_masks[i - 1][2] + 1,
            tr_masks[i - 1][1]:tr_masks[i - 1][3] + 1] = next_tr

        appr_mat = self.calculate_approximation_matrix(reduced_mats[0],
                                                       tr_masks[0])
        return appr_mat

    def calculate_approximation_matrix(self, matrix, target_region_mask):
        target_region_mask = ((target_region_mask[0], target_region_mask[1]),
                              (target_region_mask[2], target_region_mask[3]))

        U, s, VT = np.linalg.svd(matrix, full_matrices=False)

        Sigma = np.zeros((matrix.shape[0], matrix.shape[1]))
        Sigma[:min(matrix.shape[0], matrix.shape[1]),
        :min(matrix.shape[0], matrix.shape[1])] = np.diag(s)

        k = s.shape[0]
        ssim = 1
        prev_mat = matrix
        cur_mat = matrix
        init_patches = patch_extractor.get_patches(matrix, target_region_mask)

        while True:
            if ssim < 0.3:
                break
            prev_mat = cur_mat
            cur_mat = U[:, :k - 1] @ Sigma[:k - 1, :k - 1] @ VT[:k - 1, :]
            cur_mat = np.round(cur_mat)
            cur_mat = cur_mat.astype(np.uint8)

            cur_patches = patch_extractor.get_patches(cur_mat, target_region_mask)
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
