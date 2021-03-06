# CVT 

Image transforms used in deep learning training.

## installation

```
pip install cvt
```

```bash
git clone git@github.com:wangjie-ruc/cvt.git
cd cvt
python setup.py install
```

## usage

Please Refer to test folder.

Data should be put in a dict before transformed. 'image' for image data, 'mask' for pixel-wise annotation, for multi-label segmentation, mask are organised in a list. keypoint and bounding box are under developing. 

We can transform a single image directly without putting it in a dict.

```python
img = cv.imread("")
mask = cv.imread("")
data = {'image':img, 'mask':mask}
```


cfg.json
```json
{
    "sequence":{
        "sample": {
            "bright": {"scale": 0.1},
            "contrast": {"scale": 0.1},
            "hue": {"scale": 0.1}
        },
        "shuffle": {
            "vflip": {},
            "hflip": {},
            "rotate": {"degrees": 30}
        }
    }
}
```


```python
import cvt
import matplotlib.pyplot as plt

tfms = cvt.from_file('cfg.json')
img = cv.imread('image.jpg')[:,:,[2,1,0]]
data = {'image':img}
data = tfms(data)

plt.imshow(data['image'])
plt.show()
```

generate json config from python code
```python

from cvt.transform import (RandomBright, RandomContrast, RandomHorizontalFlip,
                           RandomHue, RandomRotation, RandomVerticalFlip,
                           Sample, Sequence, Shuffle)

tfms = Sequence([
    Sample([RandomBright(0.2), RandomContrast(0.2), RandomHue(0.2)]),
    Shuffle([RandomHorizontalFlip(), RandomVerticalFlip(), RandomRotation(30)])
])

print(tfms.to_json())
''' expected output
{
    "sequence": {
        "sample": {
            "bright": {
                "scale": [
                    0.8,
                    1.2
                ]
            },
            "contrast": {
                "scale": [
                    0.8,
                    1.2
                ]
            },
            "hue": {
                "scale": [
                    -0.2,
                    0.2
                ]
            }
        },
        "shuffle": {
            "hflip": {
                "p": 0.5
            },
            "vflip": {
                "p": 0.5
            },
            "rotate": {
                "degrees": [
                    -30,
                    30
                ],
                "interpolation": "nearest",
                "expand": false,
                "center": null,
                "fill": 0
            }
        }
    }
}
'''
```


## modules

- compose transforms
1. Sequence
2. Shuffle
3. Sample

- color transforms
1. RandomBright
2. RandomContrast
3. RandomSaturation
4. RandomHue
5. RandomGamma

- geometry transforms
1. RandomHorizontalFlip
2. RandomVerticalFlip
3. RandomRotation
4. Resize
5. RandomResizedCrop
6. RandomCrop
7. Pad

- tensor
1. ToTensor
2. is_numpy
3. is_pil
4. is_tensor
5. to_numpy
6. to_pil
7. to_tensor