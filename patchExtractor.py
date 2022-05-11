

class patch_extractor:
    extraction_region = ((),())

    def setDesieredRegion(self, extraction_region):
        self.extraction_region = extraction_region

    def get_patches(img, target_region):

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
            for row in range((R//divider) + 1):
                for col in range((C//divider)+1):
                    if (row*divider + divider >= R):
                        if (col*divider>= C):
                            patch = img[row*divider:row*divider+divider, col*divider:]
                        else:
                            patch = img[row*divider:row*divider+divider, col*divider:col*divider+divider]
                    else:
                        if (col*divider+divider>= C):
                            patch = img[row*divider:, col*divider:]
                        else:
                            patch = img[row*divider:, col*divider:col*divider+divider]
                    patches.append(patch)
        
        return patches


