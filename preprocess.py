#!/usr/bin/env python3
"""
Preprocesses RaFD data to accomplish two things:

    1. Trim the image of the sides, since there is little relevant information
       near the edges (i.e. the white background and black clothes).
    2. Create a segmentation mask of where the person is by creating a binary
       mask according to the background color.

Note - these parameters seem to work well:
    --trim 16
    --top 16
    --size 160 128 (or anything with a 5:4 ratio)

"""

import argparse
import os

import numpy as np
import scipy.misc as misc
from skimage.segmentation import quickshift
from skimage.morphology import binary_closing, binary_dilation, disk, square
from tqdm import tqdm


def process_image(image, trim=32, size=(192, 128), top=0):
    """
    Processes an image by doing the following:

        1. Trims the edges, getting rid of most of the white background and the
           subject's chest.
        2. Resizes the image to the desired size.

    Args:
        image (numpy.ndarray): Image to process.
    Args (optional):
        trim (int): How many pixels from the edge to trim off the top and sides.
        size (tuple<int>): Desired size of the image.
        top (int): An extra amount to trim from the top.
    """
    trim = int(trim)
    size = tuple(size)

    # Trim edges and resize

    height, width, d = image.shape

    width  = int(width-2*trim)
    height = int(width*size[0]/size[1])

    image = image[trim+top:trim+height,trim:trim+width,:]
    image = misc.imresize(image, size+(d,))

    # Segment the image

    seg = quickshift(image, kernel_size=2.5)

    # Create masks based on segments whose average color is light-gray

    def is_bg(x):
        """ Returns true if the color is light-gray-like. """
        return 176 < x.mean() and x.std() < 10

    def color_from_mask(image, mask):
        """ Gets the average color of an image from a given mask. """
        x = np.ma.masked_array(image, mask=mask)
        return x.sum(axis=0).sum(axis=0) / (np.prod(size)-mask.sum()/3)

    mask = np.zeros(image.shape[0:2])
    for i in range(0, np.max(seg)):
        seg_mask = np.tile((seg!=i).reshape(size+(1,)), (1,1,d))
        color = color_from_mask(image, seg_mask)
        if is_bg(color):
            mask = np.logical_or(mask, seg==i)
    mask = np.logical_not(mask)

    # Close and dilate a little bit to cover for any parts of the subject that
    # get accidentally masked over
    mask = binary_closing(mask, disk(2))
    mask = binary_dilation(mask, disk(2))

    # Get a masked image to preview
    preview = image * np.tile(mask.reshape(size+(1,)), (1,1,d))

    return image, 255*mask, preview


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
            "Preprocess RaFD images to use for learning.")
    parser.add_argument('-d', '--data', type=str, default='data/', help=
            "Directory where the data lives.")
    parser.add_argument('-o', '--output', type=str, default='processed/', help=
            "Directory to write output files to.")
    parser.add_argument('--trim', type=int, default=32, help=
            "The number of pixels to trim from the edges.")
    parser.add_argument('--top', type=int, default=0, help=
            "An extra amount to trim from the top")
    parser.add_argument('--size', type=int, nargs=2, default=[196, 128], help=
            "The size to resize images to.")
    args = parser.parse_args()

    # Locations in the output
    output_dirs = {
        'image'  : os.path.join(args.output, 'image'),
        'mask'   : os.path.join(args.output, 'mask'),
        'preview': os.path.join(args.output, 'preview'),
    }

    # Create output dirs if not already existing
    for directory in output_dirs.values():
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Go through each image and process
    for filename in tqdm(os.listdir(args.data)):
        filepath = os.path.join(args.data, filename)
        image = misc.imread(filepath)

        image, mask, preview = process_image(image, trim=args.trim,
                size=args.size, top=args.top)

        misc.imsave( os.path.join(output_dirs['image'], filename), image)
        misc.imsave( os.path.join(output_dirs['mask'], filename), mask)
        misc.imsave( os.path.join(output_dirs['preview'], filename), preview)


