from  PIL import Image, ImageEnhance
import numpy as np 

img  = np.random.randint(0, 255, size=(256, 256), dtype=np.uint8)
scale = 1.5

def adjust_brightness_np(img, scale):
    img = img * scale
    np.clip(img, 0, 255, out=img)
    return np.uint8(img)

def adjust_brightness_pil(img, scale):
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(scale)
    return img

print(img.mean())
img_np = adjust_brightness_np(img, scale)
img_pil = adjust_brightness_pil(Image.fromarray(img), scale)
print(img_np.mean())
print(np.array(img_pil).mean())