

class patch_extractor:
    extraction_region = ((),())

    def setDesieredRegion(self, extraction_region):
        self.extraction_region = extraction_region

    def get_patches(img, target_region):
        
        img = img.tolist()

        for row in range(len(img)):
            if row >= target_region[0][1] and row <= target_region[1][1]:
                img[row] = img[row][:target_region[0][0]] + img[row][target_region[1][0]+1:]
        
        return img 


