#!/usr/bin/env python3
"""
Learn a deconvolution network to generate and interpolate between faces.

Adapted from A. Dosovitskiy et al, "Learning to Generate Chairs, Tables, and
Cars with Convolutional Networks" 2014.

"""

import argparse
import os

import scipy.misc as misc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(decription=
            "Learn a deconvolution network to generate faces.")
    parser.add_argument('command', type=str, help=
            "Command to run. One of 'compile', 'train', or 'generate'.")

    parser.add_argument('-d', '--data', type=str, default='processed/', help=
            "Directory where the data lives.")
    parser.add_argument('-o', '--output', type=str, default='output/', help=
            "Directory to place output images.")
    parser.add_argument('-m', '--model', type=str, default='model.yaml', help=
            "Model file to load/save.")
    parser.add_argument('-w', '--weights', type=str, default='', help=
            "Weights file to load/save.")

    parser.add_argument('--include-children', action='store_true', help=
            "Include children in the dataset.")
    parser.add_argument('--use-gaze', action='store_true', help=
            "Use all gazes of each subject. (Default: Only use front-gaze)")

    args = parser.parse_args()


