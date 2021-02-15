#----------------------------------------------------------------------------
# Utilities for data augmentation.
#----------------------------------------------------------------------------

import tensorflow as tf

#----------------------------------------------------------------------------
# Utilities for image augmentation

class ImageAugmentation(tf.Module):
    def __init__(self, augs):
        """
        Generic class for image augmentation.

        Arguments:
            augs : a list of augmentation functions
        """
        self.augs = augs 

    def __call__(self, x):
        for i in range(len(self.augs)):
            x = self.augs[i](x)
        return x

def hflip(x, seed=42):
    """
    Performs a horizontal flip on the batch of images x.
    """
    return tf.image.random_flip_left_right(x, seed=seed)

def crop(x, crop_size=4, seed=42):
    """
    'Crops' the images randomly by padding it first and then extracting a random crop of the images.
    Then, the output images have the same size as the input ones.
    """
    shape = x.shape
    crop_shape = [[crop_size, crop_size], [crop_size, crop_size], [0,0]]
    if len(x.shape)==4: crop_shape = [[0,0]]+crop_shape # if it is a batch of images
    x = tf.pad(x, crop_shape)
    return tf.image.random_crop(x, shape, seed=seed)

#----------------------------------------------------------------------------
