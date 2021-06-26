import numpy as np
from albumentations.core.composition import Compose


def channel_last(rgb_image, albumentations_compose: Compose = None, **kwargs):
    # Augmentation
    if albumentations_compose:
        rgb_frame = albumentations_compose(image=rgb_image)['image']

    # Transforms
    features_dict = {'rgb_image': np.moveaxis(np.array(rgb_frame), -1, 0)}
    features_dict.update(kwargs)
    return features_dict
