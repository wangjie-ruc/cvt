import collections
import json
import os

from .transform import (Identity, LabelMap, Normalize, Pad, RandomBright,
                        RandomCLAHE, RandomContrast, RandomCrop, RandomGamma,
                        RandomHorizontalFlip, RandomHue, RandomResizedCrop,
                        RandomRotation, RandomSaturation, RandomVerticalFlip,
                        Resize, Sample, Sequence, Shuffle, ToTensor)

tfms_map = {
    'identity': Identity,
    
    'bright': RandomBright,
    'contrast': RandomContrast,
    'gamma': RandomGamma,
    'saturation': RandomSaturation,
    'hue': RandomHue,
    'clahe': RandomCLAHE,

    'hflip': RandomHorizontalFlip,
    'vflip': RandomVerticalFlip,
    'resized_crop': RandomResizedCrop,
    'resize': Resize,
    'rotate': RandomRotation,
    'crop': RandomCrop,
    'pad': Pad,

    'sequence': Sequence,
    'shuffle':  Shuffle,
    'sample': Sample,

    'totensor':  ToTensor,
    'label_map': LabelMap,
    'normalize': Normalize,
}


def _get_tfms_from_dict(cfg):
    tfms = []
    for tfm, params in cfg.items():
        assert tfm in tfms_map
        if tfm in ['sequence', 'shuffle', 'sample']:
            t = _get_tfms_from_dict(params)
            tfms.append(tfms_map[tfm](t))
            continue
        tfms.append(tfms_map[tfm](**params))
    if len(tfms) == 1 and isinstance(tfms[0], Sequence):
        return tfms[0]
    return tfms

def from_dict(cfg):
    if len(cfg) == 1 and list(cfg.keys())[0] == 'sequence':
        pass
    else:
        cfg = collections.OrderedDict([('sequence', cfg)])
    return _get_tfms_from_dict(cfg)

def from_file(filename):
    if filename.endswith('json'):
        cfg_dict = json.load(
            open(filename), object_pairs_hook=collections.OrderedDict)
        return from_dict(cfg_dict)


def from_option():
    pass


def from_yaml():
    pass
