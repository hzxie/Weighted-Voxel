#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Originally developed by Chris Choy <chrischoy@ai.stanford.edu>

import numpy as np

from PIL import Image

def image_transform(img, crop_x, crop_y, random_crop, flip, crop_loc=None, color_tint=None):
    """Takes numpy.array img"""

    # Slight translation
    if random_crop and not crop_loc:
        crop_loc = [np.random.randint(0, crop_y), np.random.randint(0, crop_x)]

    if crop_loc:
        cr, cc = crop_loc
        height, width, _ = img.shape
        img_h = height - crop_y
        img_w = width - crop_x
        img = img[cr:cr + img_h, cc:cc + img_w]

    if flip and np.random.rand() > 0.5:
        img = img[:, ::-1, ...]

    return img

def crop_center(img, new_height, new_width):
    height = img.shape[0]  # Get dimgensions
    width = img.shape[1]
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return img[top:bottom, left:right]

def add_random_color_background(img, color_range):
    r, g, b = [np.random.randint(color_range[i][0], color_range[i][1] + 1) for i in range(3)]
    if isinstance(img, Image.Image):
        img = np.array(img)

    if len(img.shape) >= 3 and img.shape[2] > 3:
        # If the image has the alpha channel, add the background
        alpha = (np.expand_dims(img[:, :, 3], axis=2) == 0).astype(np.float)
        img = img[:, :, :3]
        bg_color = np.array([[[r, g, b]]])
        img = alpha * bg_color + (1 - alpha) * img

    return img

def preprocess_image(img, config, train=True):
    # add random background
    img = add_random_color_background(img, config.TRAIN.NO_BG_COLOR_RANGE if train else
                                        config.TEST.NO_BG_COLOR_RANGE)

    # If the image has alpha channel, remove it.
    img_rgb = np.array(img)[:, :, :3].astype(np.float32)
    if train:
        t_img = image_transform(img_rgb, config.TRAIN.PAD_X, config.TRAIN.PAD_Y, \
                                    config.TRAIN.RANDOM_CROP, config.TRAIN.FLIP)
    else:
        t_img = crop_center(img_rgb, config.CONST.IMG_H, config.CONST.IMG_W)

    # Scale image
    t_img = t_img / 255.

    return t_img
