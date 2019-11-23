from  PIL import Image, ImageEnhance
import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt

img  = cv.imread('001.jpg')
# img = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
scale = 1.5

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

img_np = adjust_saturation_np(img, scale)
img_pil = adjust_saturation_pil(Image.fromarray(img), scale)
print(img_np.mean())
print(np.array(img_pil).mean())