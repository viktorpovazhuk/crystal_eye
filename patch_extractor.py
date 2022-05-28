from patch import Patch


class PatchExtractor:
    extraction_region = ((), ())

    def is_overlap(self, patch_r, target_region):
        print(patch_r)
        print(target_region)

        # If one rectangle is on left side of other
        if (patch_r[0][0] > target_region[1][0] or target_region[0][0] > patch_r[1][0]):
            return False

        # If one rectangle is above other
        if (patch_r[1][1] < target_region[0][1] or target_region[1][1] < patch_r[0][1]):
            return False

        return True

    def set_desired_region(self, extraction_region):
        self.extraction_region = extraction_region

    def get_patches(img, target_region):

        #i1, j1, i2, j2 = target_region[0], target_region[1], target_region[2], target_region[3]
        #target_region = ((j1 ,i1), (j2, i2))

        divider = 20
        patches = []

        im1 = img[:target_region[0][1], :]
        im2 = img[target_region[0][1]:target_region[1][1], : target_region[0][0]]
        im3 = img[target_region[0][1]:target_region[1][1], target_region[1][0]:]
        im4 = img[target_region[1][1]:, :]

        images = [im1, im2, im3, im4]

        for img in images:
            R = img.shape[0]
            C = img.shape[1]
            for row in range((R // divider) + 1):
                for col in range((C // divider) + 1):
                    if (row * divider + divider >= R):
                        if (col * divider >= C):
                            patch = img[row * divider:row * divider + divider,
                                    col * divider:]
                        else:
                            patch = img[row * divider:row * divider + divider,
                                    col * divider:col * divider + divider]
                    else:
                        if (col * divider + divider >= C):
                            patch = img[row * divider:, col * divider:]
                        else:
                            patch = img[row * divider:,
                                    col * divider:col * divider + divider]
                    patches.append(patch)

        return patches

    def get_patches_by_size(self, img, x_scale, y_scale, size,
                            target_region=extraction_region):
        
        #i1, j1, i2, j2 = target_region[0], target_region[1], target_region[2], target_region[3]
        #target_region = ((j1 ,i1), (j2, i2))

        final_list = []

        y_max, x_max = img.shape[0], img.shape[1]

        for i in range(0, y_max - size, y_scale):

            if i + size + y_scale >= y_max:
                i = y_max - size
                for j in range(0, x_max - size, x_scale):
                    if j + size + x_scale >= x_max:
                        if not self.is_overlap(
                                ((x_max - size, i), (x_max, i + size)),
                                target_region):
                            final_list.append(Patch((i, x_max - size),
                                                    img[i:i + size, x_max - size:x_max]))
                    else:
                        if not self.is_overlap(((j, i), (j + size, i + size)),
                                            target_region):
                            final_list.append(Patch((i, j), img[i:i + size, j:j + size]))

            else:
                for j in range(0, x_max - size, x_scale):
                    if j + size + x_scale >= x_max:
                        if not self.is_overlap(((i, x_max - size), (x_max, i + size)),
                                            target_region):
                            final_list.append(Patch((x_max - size, i),
                                                    img[i:i + size, x_max - size:x_max]))
                    else:
                        if not self.is_overlap(((i, j), (j + size, i + size)), 
                                                target_region):
                            final_list.append(Patch((j, i), img[i:i + size, j:j + size]))


        return final_list
