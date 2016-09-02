"""
Instance class to hold data for each example.

"""

import os
import random

from keras import backend as K

import numpy as np
import scipy.misc as misc
from tqdm import tqdm


# ---- Enum classes for vector descriptions

class Emotion:
    angry         = [1., 0., 0., 0., 0., 0., 0., 0.]
    contemptuous  = [0., 1., 0., 0., 0., 0., 0., 0.]
    disgusted     = [0., 0., 1., 0., 0., 0., 0., 0.]
    fearful       = [0., 0., 0., 1., 0., 0., 0., 0.]
    happy         = [0., 0., 0., 0., 1., 0., 0., 0.]
    neutral       = [0., 0., 0., 0., 0., 1., 0., 0.]
    sad           = [0., 0., 0., 0., 0., 0., 1., 0.]
    surprised     = [0., 0., 0., 0., 0., 0., 0., 1.]

    @classmethod
    def length(cls):
        return len(Emotion.neutral)


# ---- Loading functions

class RaFDInstances:

    def __init__(self, directory):
        """
        Constructor for a RaFDInstances object.

        Args:
            directory (str): Directory where the data lives.
            image_size (tuple<int>): Size images should be cropped and resized to.
        """

        self.directory = directory

        # A list of all files in the current directory (no kids, only frontal gaze)
        self.filenames = [x for x in os.listdir(directory)
                if 'Kid' not in x and 'frontal' in x]

        # The number of times the directory has been read over
        self.num_iterations = 0

        # Count identities and map each identity present to a contiguous value
        identities = list()
        for filename in self.filenames:
            identity = int(filename.split('_')[1])-1 # Identities are 1-indexed
            if identity not in identities:
                identities.append(identity)
        self.identity_map = dict()
        for idx, identity in enumerate(identities):
            self.identity_map[identity] = idx

        self.num_identities = len(self.identity_map)
        self.num_instances = len(self.filenames)


    def load_data(self, image_size, verbose=False):
        """
        Loads RaFD data for training.

        Args:
            image_size (tuple<int>): Size images should be resized to.
        Returns:
            numpy.ndarray, training data (face parameters).
            numpy.ndarray, output data (the actual images to generate).
        """

        inputs = {
            'emotion'    : np.empty((self.num_instances, len(Emotion.neutral))),
            'identity'   : np.empty((self.num_instances, self.num_identities)),
            'orientation': np.empty((self.num_instances, 2)),
        }

        if K.image_dim_ordering() == 'th':
            outputs = np.empty((self.num_instances, 3)+image_size)
        else:
            outputs = np.empty((self.num_instances,)+image_size+(3,))

        all_instances = range(0, len(self.filenames))
        if verbose:
            all_instances = tqdm(all_instances)

        for i in all_instances:
            instance = RaFDInstance(self.directory, self.filenames[i], image_size)

            inputs['emotion'][i,:] = instance.emotion
            inputs['identity'][i,:] = instance.identity_vector(self.identity_map)
            inputs['orientation'][i,:] = instance.orientation

            if K.image_dim_ordering() == 'th':
                outputs[i,:,:,:] = instance.th_image()
            else:
                outputs[i,:,:,:] = instance.tf_image()

        return inputs, outputs


# ---- Instance class definition

class RaFDInstance:
    """
    Holds information about each example.
    """

    def __init__(self, directory, filename, image_size, trim=24, top=24):
        """
        Constructor for an Instance object.

        Args:
            directory (str): Base directory where the example lives.
            filename (str): The name of the file of the example.
            image_size (tuple<int>): Size to resize the imagae to.
        Args (optional):
            trim (int): How many pixels from the edge to trim off the top and sides.
            top (int): How much extra to trim off the top.
        """

        self.image = misc.imread( os.path.join(directory, filename) )

        # Trim the image to get more of the face

        height, width, d = self.image.shape

        width = int(width-2*trim)
        height = int(width*image_size[0]/image_size[1])

        self.image = self.image[trim+top:trim+height,trim:trim+width,:]

        # Resize and fit between 0-1
        self.image = misc.imresize( self.image, image_size )
        self.image = self.image / 255.0

        #self.mask  = misc.imread( os.path.join(directory, 'mask',  filename) )
        #self.mask  = misc.imresize( self.mask, image_size )
        #self.mask  = self.mask / 255.0

        # Parse filename to get parameters

        items = filename.split('_')

        # Represent orientation as sin/cos vector
        angle = np.deg2rad(float(items[0][-3:])-90)
        self.orientation = np.array([np.sin(angle), np.cos(angle)])

        self.identity_index = int(items[1])-1 # Identities are 1-indexed

        self.emotion = np.array(getattr(Emotion, items[4]))


    def identity_vector(self, identity_map):
        """
        Creates a one-in-k encoding of the instance's identity.

        Args:
            identity_map (dict): Mapping from identity to a unique index.
        Returns:
            numpy.ndarray, the identity vector.
        """

        identity_vec = np.zeros(len(identity_map), dtype=np.float32)
        identity_vec[ identity_map[self.identity_index] ] = 1.

        return identity_vec


    def th_image(self):
        """
        Returns a Theano-ordered representation of the image.
        """

        image = np.empty((3,)+self.image.shape[0:2])
        for i in range(0, 3):
            image[i,:,:] = self.image[:,:,i]
        return image


    def tf_image(self):
        """
        Returns a TensorFlow-ordered representation of the image.
        """

        # As-is
        return self.image


