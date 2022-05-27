from patch import Patch


class patch_extractor:
    extraction_region = ((),())

    def is_overlap(img,  target_region = extraction_region):
        # If one rectangle is on left side of other
        if(img[0][0] > target_region[1][0] or target_region[0][0] > img[1][0]):
            return False
    
        # If one rectangle is above other
        if(img[1][1] < target_region[0][1] or target_region[1][1] < img[0][1]):
            return False
    
        return True    

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

    def get_patches_by_size(self, img, x_scale, y_scale, size, target_region = extraction_region):
        
        final_list = []

        y_max, x_max = img.shape[0], img.shape[1]

        for i in range(0, y_max - size, y_scale):
            for j in range(0, x_max - size, x_scale):
                if not self.is_overlap(((j, i), (j+size, i+size)), target_region):
                    final_list.append(Patch((j, i), img[i:i+size, j:j+size]))
                if j+size+x_scale >= y_max:
                    if not self.is_overlap(((x_max-size, i), (x_max, i+size)), target_region):
                        final_list.append(Patch((x_max-size, i), img[i:i+size, x_max-size:x_max]))
            if i+size+y_scale >= y_max:
                i = y_max - size
                for j in range(0, x_max - size, x_scale):
                    if not self.is_overlap(((j, i), (j+size, i+size)), target_region):
                        final_list.append(Patch((j, i), img[i:i+size, j:j+size]))
                    if j+size+x_scale >= y_max:
                        if not self.is_overlap(((x_max-size, i), (x_max, i+size)), target_region):
                            final_list.append(Patch((x_max-size, i) ,img[i:i+size, x_max-size:x_max]))
        
        return final_list

