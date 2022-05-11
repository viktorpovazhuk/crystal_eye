import numpy as np


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

        U, s, VT = np.linalg.svd(mat)

        Sigma = np.zeros((mat.shape[0], mat.shape[1]))
        Sigma[:min(mat.shape[0], mat.shape[1]),
        :min(mat.shape[0], mat.shape[1])] = np.diag(
            s)

        print(U, Sigma, VT)

        k = len(s)
        ssim = 1
        prev_mat = mat
        cur_mat = mat
        init_patches = patch_extractor.get_patches(mat, target_region_mask)

        while True:
            if ssim < 0.9:
                break
            prev_mat = cur_mat
            cur_mat = U[:, :k - 1] @ Sigma[:k - 1, :k - 1] @ VT[:k - 1, :]
            cur_patches = patch_extractor.get_patches(cur_mat, target_region_mask)
            ssim = self.calculate_ssim(init_patches, cur_patches)
            k -= 1

        return prev_mat

    def calculate_ssim(self, initial_patches, current_patches):
        c1, c2 = 1, 5
        n = len(initial_patches)
        ssim = 0

        i = 0
        while i < n:
            x = initial_patches[i] 
            y = current_patches[i]
        #for (x, y) in (initial_patches, current_patches):
            ssim += (((2 * np.mean(x) * np.mean(y) + c1) * (2 * np.cov(x, y)[0,1] + c2)) /
                     ((np.mean(x) ** 2 + np.mean(y) ** 2 + c1) * (np.std(x) ** 2 + np.std(y) ** 2 + c2)))
            i += 1
        ssim = ssim / n
        return ssim
