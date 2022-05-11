

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
            for row in range(0,(R//divider) + 1, divider):
                for col in range(0, (C//divider)+1, divider):
                    if (row*divider >= R):
                        if (col*divider>= C):
                            patch = img[row:row+divider, col:]
                        else:
                            patch = img[row:row+divider, col:col+divider]
                    else:
                        if (col*divider>= C):
                            patch = img[row:, col:]
                        else:
                            patch = img[row:, col:col+divider]
                    patches.append(patch)
        
        return patches


