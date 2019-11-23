from  PIL import Image, ImageEnhance
import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt

img  = cv.imread('001.jpg')
# img = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
scale = 0.5

def adjust_brightness_np(img, scale):
    img = img * scale
    np.clip(img, 0, 255, out=img)
    return np.uint8(img)

def adjust_brightness_pil(img, scale):
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(scale)
    return img


def to_gray(img, keep_dim=False):
    r, g, b = cv.split(img)
    gray = r * 0.299 + g * 0.587 + b * 0.114
    np.clip(gray, 0, 255, gray)
    gray = np.uint8(gray)
    if keep_dim:
        return cv.merge([gray] * 3)
    return gray

def adjust_saturation_np(img, scale):
    gray = to_gray(img, keep_dim=True)
    img = np.int32(img)
    img = img * scale + gray * (1.0 - scale) 
    np.clip(img, 0, 255, out=img)
    return np.uint8(img)

def adjust_saturation_pil(img, scale):
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(scale)
    return img

def adjust_contrast_np(img, scale):
    mean = round(to_gray(img).mean())
    gray = np.ones_like(img, dtype=np.uint8) * int(mean)
    img = img * scale + gray * (1.0 - scale) 
    np.clip(img, 0, 255, out=img)
    return np.uint8(img)


def adjust_contrast_pil(img, scale):
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(scale)
    return img


def adjust_hue_np(img, scale):

    if not(-0.5 <= scale <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(scale))
    img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    h, s, v = cv.split(img)

    h = np.int32(h)
    h = np.uint8((h + (scale * 180) % 181) % 181)

    img = cv.merge([h, s, v])
    img = cv.cvtColor(img, cv.COLOR_HSV2RGB)
    return img

def adjust_hue_pil(img, scale):

    if not(-0.5 <= scale <= 0.5):
        raise ValueError('scale is not in [-0.5, 0.5].'.format(scale))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(scale * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img

img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
img_pil_hsv = np.array(Image.fromarray(img).convert('HSV'))
print(img_hsv[:,:,0].max())
print(img_pil_hsv[:,:,0].max())

img_np = adjust_hue_np(img, scale)
img_pil = adjust_hue_pil(Image.fromarray(img), scale)
print(img_np.mean())
print(np.array(img_pil).mean())

from __future__ import division
import torch
import sys
import math
from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
from numpy import sin, cos, tan
import numbers
import collections
import warnings

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy(img):
    return isinstance(img, np.ndarray)


def _is_numpy_image(img):
    return img.ndim in {2, 3}


def to_tensor(pic):


    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        print('255 div')
        return img.float().div(255)
    else:
        return img
