import torch
import torchvision

import pandas as pd
import numpy as np

from PIL import Image


class LungDataset(torch.utils.data.Dataset):
    def __init__(self, origin_mask_list, origins_folder, masks_folder, transforms=None):
        self.origin_mask_list = origin_mask_list
        self.origins_folder = origins_folder
        self.masks_folder = masks_folder
        self.transforms = transforms
    
    def __getitem__(self, idx):
        origin_name, mask_name = self.origin_mask_list[idx]
        # changed from convert("P") to "L" fixed images being loaded as negatives which killed prediction
        origin = Image.open(self.origins_folder / (origin_name + ".png")).convert("L")
        #print("getitem: ", self.origins_folder, origin_name, np.array(origin).shape, np.array(origin)[50,50])
        mask = Image.open(self.masks_folder / (mask_name + ".png")).convert("L")
        if np.array(mask).shape == (512,512):
            origin = torchvision.transforms.functional.resize(origin, (512,512))
        if self.transforms is not None:
            origin, mask = self.transforms((origin, mask))
            
        #print("getitem: ", np.array(origin)[0,0])
        origin = torchvision.transforms.functional.to_tensor(origin) - 0.5
        #print(origin.min(), origin.max())
        mask = np.array(mask)
        #if mask.max() == 1: # our masks are not [0,255] but float [0,1], so binary convert
        #   print("ichwarhier")
        #    mask = (torch.tensor(mask).mul(255) > 128).long()
        #else:
        #    print("diesichwarhier")
        mask = (torch.tensor(mask) > 128).long() 
        return origin, mask
        
    
    def __len__(self):
        return len(self.origin_mask_list)

    
class Pad():
    def __init__(self, max_padding):
        self.max_padding = max_padding
        
    def __call__(self, sample):
        origin, mask = sample
        padding = np.random.randint(0, self.max_padding)
#         origin = torchvision.transforms.functional.pad(origin, padding=padding, padding_mode="symmetric")
        origin = torchvision.transforms.functional.pad(origin, padding=padding, fill=0)
        mask = torchvision.transforms.functional.pad(mask, padding=padding, fill=0)
        return origin, mask

class Rotate():
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
    
    def __call__(self, sample):
        origin, mask = sample
        angle = np.random.randint(self.lower, self.upper)
        origin = torchvision.transforms.functional.rotate(origin, angle)
        mask = torchvision.transforms.functional.rotate(mask, angle)
        return origin, mask



class Crop():
    def __init__(self, max_shift):
        self.max_shift = max_shift
        
    def __call__(self, sample):
        origin, mask = sample
        tl_shift = np.random.randint(0, self.max_shift)
        br_shift = np.random.randint(0, self.max_shift)
        origin_w, origin_h = origin.size
        crop_w = origin_w - tl_shift - br_shift
        crop_h = origin_h - tl_shift - br_shift
        
        origin = torchvision.transforms.functional.crop(origin, tl_shift, tl_shift,
                                                        crop_h, crop_w)
        mask = torchvision.transforms.functional.crop(mask, tl_shift, tl_shift,
                                                        crop_h, crop_w)
        return origin, mask


class Resize():
    def __init__(self, output_size):
        self.output_size = output_size
        
    def __call__(self, sample):
        origin, mask = sample
        origin = torchvision.transforms.functional.resize(origin, self.output_size)
        mask = torchvision.transforms.functional.resize(mask, self.output_size)
        
        return origin, mask


def blend(origin, mask1=None, mask2=None):
    #print("Blend raw: ", origin[0].shape, origin.min(), origin.max())
    #print("Blend middle pix val: ", origin[0][256,256])
    img = torchvision.transforms.functional.to_pil_image(origin + 0.5).convert("RGB")
    #print("Blend rgb image: ", np.array(img)[0,0])
    #print(np.array(img).max())
    if mask1 is not None:
        mask1 =  torchvision.transforms.functional.to_pil_image(torch.cat([
            torch.zeros_like(origin),
            torch.stack([mask1.float()]),
            torch.zeros_like(origin)
        ]))
        img = Image.blend(img, mask1, 0.2)
        
    if mask2 is not None:
        mask2 =  torchvision.transforms.functional.to_pil_image(torch.cat([
            torch.stack([mask2.float()]),
            torch.zeros_like(origin),
            torch.zeros_like(origin)
        ]))
        img = Image.blend(img, mask2, 0.2)
    
    return img
