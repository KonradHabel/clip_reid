import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2
import numpy as np

class RectResize(ImageOnlyTransform):
    def __init__(self,
                 size, 
                 padding_value=0,
                 interpolate=cv2.INTER_LINEAR_EXACT,
                 always_apply=False,
                 p=1.0):
        
        super(RectResize, self).__init__(always_apply, p)
        self.size = size
        self.padding_value = padding_value
        self.interpolate = interpolate

    def apply(self, image, **params):

        h, w, c = image.shape
        
        if w == h:
            img_resize = cv2.resize(image, dsize=(self.size[1], self.size[0]), interpolation = self.interpolate)
            return img_resize
        
        else:
            img_pad = np.full((self.size[0], self.size[1], c), self.padding_value, dtype=np.uint8)
            if h > w:
                resize_value = self.size[0] / h
                w_new = round(w * resize_value)
                img_resize = cv2.resize(image, dsize=(w_new, self.size[0]), interpolation = self.interpolate)
                padding = (self.size[1] - w_new) // 2
                img_pad[:, padding:padding+w_new,:] = img_resize
            else:
                resize_value = self.size[1] / w
                h_new = round(h * resize_value)
                img_resize = cv2.resize(image, dsize=(self.size[1], h_new), interpolation = self.interpolate)
                padding = (self.size[0] - h_new) // 2
                img_pad[padding:padding+h_new,:,:] = img_resize
            return img_pad
            
    def get_transform_init_args_names(self):
        return ("size", "padding_value", "interpolate")
    

def get_transforms(image_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
    
    # only use zero padding for resize
    val_transforms = A.Compose([RectResize(size=image_size, padding_value=0, interpolate=cv2.INTER_LINEAR_EXACT, p=1.0),
                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])
 
    # feel free to add more augmentations for train transforms
    # random horizontal flip is applied in the dataloader on query and gallery simultaneously
    train_transforms = A.Compose([RectResize(size=image_size, padding_value=0, interpolate=cv2.INTER_LINEAR_EXACT, p=1.0),
                                  A.Normalize(mean, std),
                                  ToTensorV2(),
                                ])
    
    return val_transforms, train_transforms