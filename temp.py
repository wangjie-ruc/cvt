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