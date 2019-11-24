from cvt.transform import (RandomBright, RandomContrast, RandomHorizontalFlip,
                           RandomHue, RandomRotation, RandomVerticalFlip,
                           Sample, Sequence, Shuffle)

tfms = Sequence([
    Sample([RandomBright(0.2), RandomContrast(0.2), RandomHue(0.2)]),
    Shuffle([RandomHorizontalFlip(), RandomVerticalFlip(), RandomRotation(30)])
])

