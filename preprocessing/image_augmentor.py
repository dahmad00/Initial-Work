import os
import imageio.v2 as imageio
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug import augmenters as iaa

class augmentor:
    def __init__(self):
        self.path = None
        self.images = None
        self.imgs_aug = None
        pass

    def load(self, path):
        self.path = path
        self.images = []
        img_names = list(os.listdir(path))

        count = 0
        for img_name in img_names:
            img = imageio.imread(img_path + img_name)
            count += 1
            self.images.append(img)
            if count % 100 == 0:
                print(str(count) + " images read")

    def rotate(self, range = (360,-360), img_resize = (None,None)):

        if img_resize[0] != None:
            resize = iaa.Resize({"height":100, "width":100})
        
        rotate = iaa.Affine(rotate=(-360, 360)) 
        self.imgs_aug = []
        
        count = 0
        for i in range(1000):
            
            if img_resize != None:
                aug = resize(image = self.images[i])
            else:
                aug = self.images[i]

            aug = rotate(image = aug)
            self.imgs_aug.append(aug)
            count += 1

            if count % 100 == 0:
               print(str(count) + " images rotated")

        return self.imgs_aug

aug = augmentor()
aug.load(r'F:\FYP\Initial\apples\train')

rotated = aug.rotate(img_resize = [200,200])


        