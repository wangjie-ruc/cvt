import torch 
import numpy as np 
from PIL import Image

def is_numpy(img):
    return isinstance(img, np.ndarray)

def is_pil(img):
    return isinstance(img, Image.Image)

def is_tensor(img):
    return isinstance(img, torch.Tensor)

def to_numpy(img):
    if is_tensor(img):
        return img.detach().cpu().numpy()
    return np.array(img)

def to_pil(img):
    if is_tensor(img):
        img = to_numpy(img)
    if is_numpy(img):
        return Image.fromarray(np.uint8(img))
    return img
