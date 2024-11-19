import os
import numpy as np
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob

class CAMUSBase(Dataset):
    def __init__(self,
                 data_root,
                 isvalid=False,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.isvalid = isvalid
        self.data_root = data_root
        self.allimg_paths = glob.glob(os.path.join(data_root, '*.npy'))
        self.allimg_paths.sort()

        if not isvalid:
            self.image_paths = self.allimg_paths
        else:
            self.image_paths = self.allimg_paths[int(len(self.allimg_paths)*0.99):]

        self._length = len(self.image_paths)

        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = np.load(self.image_paths[i])
        #image = self.flip(image)
        iterval = np.max(image) - np.min(image)
        minval = np.min(image)
        image = (image-minval) / iterval
        image = self.flip(torch.from_numpy(image))
        #example["image"] = ((image-np.min(image)) / iterval).astype(np.float32)
        example["image"] = image
        return example


class CAMUSTrain(CAMUSBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="your/training/image/saving/path", isvalid=False, **kwargs)


class CAMUSValidation(CAMUSBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="your/val/image/saving/path", isvalid=True, **kwargs)

class ECHOTrain(CAMUSBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="your/training/image/saving/path", isvalid=False, **kwargs)

class ECHOValidation(CAMUSBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="your/val/image/saving/path", isvalid=True, **kwargs)

        

class ACDCTrain(CAMUSBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="your/training/image/saving/path", isvalid=False, **kwargs)

class ACDCValidation(CAMUSBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="your/val/image/saving/path", isvalid=True, **kwargs)