import numpy as np


def reduce_gaussian_mask_2d(window=5, s=1):
    idx = np.linspace(-(window - 1) // 2, window // 2, num=window)
    mask_1d = np.exp((-idx ** 2 / (2 * s ** 2)))
    mask = np.outer(mask_1d, mask_1d)
    mask = mask / np.sum(mask)
    return mask


def expand_gaussian_mask_2d(i, j, window=5, s=1):
    even_idx = np.array(
        [p for p in range(-(window - 1) // 2, window // 2 + 1) if p % 2 == 0])
    odd_idx = np.array(
        [p for p in range(-(window - 1) // 2, window // 2 + 1) if p % 2 == 1])
    even_mask_1d = np.exp((-even_idx ** 2 / (2 * s ** 2)))
    odd_mask_1d = np.exp((-odd_idx ** 2 / (2 * s ** 2)))
    if i % 2 == 0:
        row_mask = even_mask_1d
    else:
        row_mask = odd_mask_1d
    if j % 2 == 0:
        col_mask = even_mask_1d
    else:
        col_mask = odd_mask_1d
    mask = np.outer(row_mask, col_mask)
    mask = mask / np.sum(mask)
    return mask


def reduce_matrix(matrix, window=5):
    m, n = matrix.shape
    reduced_mat = np.zeros((m // 2, n // 2))
    mask = reduce_gaussian_mask_2d()

    for i in range(m // 2):
        for j in range(n // 2):
            block = matrix[
                    max(2 * i - window // 2, 0):min(2 * i + window // 2 + 1, m - 1),
                    max(2 * j - window // 2, 0):min(2 * j + window // 2 + 1, n - 1)]
            block = np.pad(block,
                           ((0, window - block.shape[0]),
                            (0, window - block.shape[1])),
                           'constant')
            pixel_val = np.dot(block.reshape(np.product(block.shape)),
                               mask.reshape(np.product(mask.shape)))
            reduced_mat[i, j] = pixel_val

    reduced_mat = reduced_mat.astype(np.uint8)

    return reduced_mat


def expand_matrix(matrix, window=5):
    m, n = matrix.shape[0], matrix.shape[1]
    expanded_mat = np.zeros((m * 2, n * 2))
    masks = {(0, 0): expand_gaussian_mask_2d(0, 0),
             (0, 1): expand_gaussian_mask_2d(0, 1),
             (1, 0): expand_gaussian_mask_2d(1, 0),
             (1, 1): expand_gaussian_mask_2d(1, 1)}

    for i in range(m * 2):
        for j in range(n * 2):
            if i % 2 == 0:
                row_offset = 2
            else:
                row_offset = 1
            if j % 2 == 0:
                col_offset = 2
            else:
                col_offset = 1
            block = matrix[
                    max((i - row_offset) // 2, 0):min((i + row_offset) // 2,
                                                      m - 1) + 1,
                    max((j - col_offset) // 2, 0):min((j + col_offset) // 2,
                                                      n - 1) + 1]
            mask = masks[(i % 2, j % 2)]
            if mask.shape[0] != block.shape[0] or mask.shape[1] != block.shape[1]:
                block = np.pad(block,
                               ((0, mask.shape[0] - block.shape[0]),
                                (0, mask.shape[1] - block.shape[1])),
                               'constant')
            pixel_val = np.dot(block.reshape(np.product(block.shape)),
                               mask.reshape(np.product(mask.shape)))
            expanded_mat[i, j] = pixel_val

    expanded_mat = expanded_mat.astype(np.uint8)

    return expanded_mat
