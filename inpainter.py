import numpy as np
from matplotlib import pyplot as plt
import sys
import math
import cv2 as cv

from patch_extractor import PatchExtractor
from scale import reduce_matrix, expand_matrix, expand_color_matrix
from skimage.restoration import inpaint

DEBUG = True


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
        harmonic_mask = np.zeros(scaled_mats[2].shape[:-1], dtype=bool)
        harmonic_mask[tr_coords[2][0]:tr_coords[2][2],
        tr_coords[2][1]:tr_coords[2][3]] = 1

        plt.ylim(400)
        plt.xlim(500)
        plt.imshow(scaled_mats[2])
        plt.show()

        # TODO: why bad result
        harmonic_mat = inpaint.inpaint_biharmonic(scaled_mats[2], harmonic_mask,
                                                  channel_axis=-1)
        plt.imshow(harmonic_mat)
        plt.show()
        harmonic_mat = harmonic_mat * 255
        harmonic_mat = harmonic_mat.astype(np.uint8)

        # TODO: convert to grayscale for harmonic inpainting -> no need
        plt.imshow(harmonic_mat)
        plt.show()

        # inpaint l2 = approximation matrix + patches
        cur_mat = self.inpaint(harmonic_mat, tr_coords[2])

        if DEBUG:
            plt.imshow(cur_mat)
            plt.show()

        for i in [1, 0]:
            # make x2
            adv_mat = expand_color_matrix(cur_mat)

            if DEBUG:
                plt.imshow(adv_mat)
                plt.show()

            # extract w + insert in next l
            adv_tr = adv_mat[tr_coords[i][0]:tr_coords[i][2],
                     tr_coords[i][1]:tr_coords[i][3]]
            cur_mat = scaled_mats[i]
            cur_mat[tr_coords[i][0]:tr_coords[i][2],
            tr_coords[i][1]:tr_coords[i][3]] = adv_tr
            # inpaint next l
            cur_mat = self.inpaint(cur_mat, tr_coords[i])

            if DEBUG:
                plt.imshow(cur_mat)
                plt.show()

        return cur_mat

    def inpaint(self, matrix, target_region_coords):
        matrix = matrix.copy()
        p_size = 4
        x_step, y_step = 2, 2
        # convert to grayscale
        gray_matrix = cv.cvtColor(matrix, cv.COLOR_BGR2GRAY)
        # find appr matrix
        appr_mat = self.calculate_approximation_matrix(gray_matrix,
                                                       target_region_coords)

        if DEBUG:
            plt.xlabel('appr_mat')
            plt.imshow(appr_mat, cmap='gray')
            plt.show()

        # extract patches from S
        patch_extractor = PatchExtractor()
        S_ps = patch_extractor.get_patches_by_size(appr_mat, x_step, y_step, p_size,
                                                   target_region_coords)
        mat_p_i, mat_p_j = target_region_coords[0], target_region_coords[1]
        print('patches extracted')
        while mat_p_i < target_region_coords[2]:
            while mat_p_j < target_region_coords[3]:
                # extract p
                W_p = appr_mat[mat_p_i:mat_p_i + p_size, mat_p_j:mat_p_j + p_size]
                # compare with all S patches
                compared = [(S_p, self.calculate_ssim([S_p.patch], [W_p])) for S_p in
                            S_ps]
                compared = sorted(compared, key=lambda x: x[1], reverse=True)
                # select first + prop values
                p_coords = compared[0][0].coordinate
                # print(matrix[p_coords[0]:p_coords[0] + p_size,
                #       p_coords[1]:p_coords[1] + p_size].shape)
                # print(appr_mat.shape)
                matrix[mat_p_i:mat_p_i + p_size, mat_p_j:mat_p_j + p_size] \
                    = matrix[p_coords[0]:p_coords[0] + p_size,
                      p_coords[1]:p_coords[1] + p_size]
                mat_p_j += p_size
            mat_p_i += p_size
            mat_p_j = target_region_coords[1]
            if mat_p_i > target_region_coords[0] + (target_region_coords[2] - target_region_coords[0]) / 2:
                plt.imshow(matrix)
                plt.show()
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
            if ssim < 0.3:
                break
            prev_mat = cur_mat
            cur_mat = U[:, :k - 1] @ Sigma[:k - 1, :k - 1] @ VT[:k - 1, :]
            cur_mat = np.round(cur_mat)
            cur_mat = cur_mat.astype(np.uint8)

            cur_patches = PatchExtractor.get_patches(cur_mat, target_region_coords)
            ssim = self.calculate_ssim(init_patches, cur_patches)
            k -= 1

            # if DEBUG:
            #     if k % 1 == 0:
            #         cv.imshow(f'SV: {k - 1}, ssim : {ssim}', cur_mat)
            #         cv.waitKey(0)
            #     pass

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
