import numpy as np
from matplotlib import pyplot as plt

from patchExtractor import patch_extractor
import sys
import math
import cv2 as cv

DEBUG = False
DEBUG_K = False


class Inpainter:
    def __init__(self, img, target_region_mask):
        self._image = img
        self._target_region_mask = target_region_mask

    def inpaint(self):
        approximation_matrix = self.calculate_approximation_matrix()
        return approximation_matrix

    def calculate_approximation_matrix(self):
        mat = self._image
        target_region_mask = self._target_region_mask

        U, s, VT = np.linalg.svd(mat, full_matrices=False)

        Sigma = np.zeros((mat.shape[0], mat.shape[1]))
        Sigma[:min(mat.shape[0], mat.shape[1]),
        :min(mat.shape[0], mat.shape[1])] = np.diag(s)

        k = s.shape[0]
        ssim = 1
        prev_mat = mat
        cur_mat = mat
        init_patches = patch_extractor.get_patches(mat, target_region_mask)

        if DEBUG_K:
            divider = 20
            plt_num = k // divider

            fig, axs = plt.subplots(int((plt_num + 1) / 2), 2, figsize=(50, 50))
            plt.subplots_adjust(wspace=0.3, hspace=0.2)

            for i in range(plt_num):
                j = (i + 1) * divider

                deb_mat = U[:, :j] @ Sigma[:j, :j] @ VT[:j, :]

                axs[i // 2][i % 2].imshow(deb_mat, cmap='gray')
                axs[i // 2][i % 2].set_title(f'j: {j}')

            fig.savefig('data/multiple_plots.png')

            return

        if DEBUG:
            # res_mat = U[:, :k] @ Sigma[:k, :k] @ VT[:k, :]
            # res_mat = np.round(res_mat)
            # res_mat = res_mat.astype(np.uint8)
            #
            # cv.imshow(f'restored', res_mat)
            # cv.waitKey(0)
            pass

        while True:
            if ssim < 0.9:
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

            if DEBUG:
                # print(f'means: {np.mean(x)}, {np.mean(y)}, cov: {np.cov(x, y)[0][1]}, '
                #       f'stds: {np.std(x)}, {np.std(y)}, ssim: {cur_ssim}\n'
                #       f'--------------------------')

                # if math.isnan(cur_ssim):
                #     print(f'{x}\n {y}\n'
                #           f'---------\n')
                pass

            ssim += cur_ssim
            i += 1
        ssim = ssim / n
        return ssim
