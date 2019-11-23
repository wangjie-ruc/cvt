from PIL import Image, ImageOps, ImageEnhance


def hflip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def vflip(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def crop(img, i, j, h, w):
    return img.crop((j, i, j + w, i + h))


def resize(img, size, interpolation=Image.BILINEAR):
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
    else:
        oh, ow = size
    return img.resize((ow, oh), interpolation)


def rotate(img, degree, interpolation=Image.BILINEAR, expand=False, center=None, fill=0):
    return img.rotate(degree, interpolation, expand, center, fillcolor=fill)

def resized_crop(img, i, j, h, w, size, interpolation=Image.BILINEAR):
    img = crop(img, i, j, h, w)
    img = resize(img, size, interpolation)
    return img

def adjust_brightness(img, scale):
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(scale)
    return img

def adjust_contrast(img, scale):
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(scale)
    return img

def adjust_saturation(img, scale):
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(scale)
    return img


def adjust_hue(img, scale):

    if not(-0.5 <= scale <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(scale))

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


def adjust_gamma(img, gamma, gain=1):
    input_mode = img.mode
    img = img.convert('RGB')

    gamma_map = [255 * gain * pow(ele / 255., gamma) for ele in range(256)] * 3
    img = img.point(gamma_map) 
    img = img.convert(input_mode)
    return img