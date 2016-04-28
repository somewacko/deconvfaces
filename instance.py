"""
Instance class to hold data for each example.

"""

import os

import numpy as np
import scipy.misc as misc
from tqdm import tqdm


# ---- Enum classes for vector descriptions

class Gender:
    female = [1., 0.]
    male   = [0., 1.]

class Emotion:
    angry         = [1., 0., 0., 0., 0., 0., 0., 0.]
    contemptuous  = [0., 1., 0., 0., 0., 0., 0., 0.]
    disgusted     = [0., 0., 1., 0., 0., 0., 0., 0.]
    fearful       = [0., 0., 0., 1., 0., 0., 0., 0.]
    happy         = [0., 0., 0., 0., 1., 0., 0., 0.]
    neutral       = [0., 0., 0., 0., 0., 1., 0., 0.]
    sad           = [0., 0., 0., 0., 0., 0., 1., 0.]
    surprised     = [0., 0., 0., 0., 0., 0., 0., 1.]


# ---- Loading functions

def load_data(directory):
    """
    Loads instances from a directory.

    Args:
        directory (str): Directory where the data lives.
    """

    identities = list()
    instances = list()

    # Load instances

    print("Loading instances...")

    for filename in tqdm(os.listdir(os.path.join(directory, 'image'))):
        # Skip kids and left/right gazes
        if 'Kid' in filename or 'frontal' not in filename:
            continue

        instance = Instance(directory, filename)

        if instance.identity not in identities:
            identities.append(instance.identity)
        instances.append(instance)

    # Normalize identities and create vectors

    identity_map = dict()
    for idx, identity in enumerate(identities):
        identity_map[identity] = idx

    for instance in instances:
        instance.create_identity_vector(identity_map)

    print("Loaded {} instances with {} identities"
            .format(len(instances), len(identity_map)))

    return instances


# ---- Instance class definition

class Instance:
    """
    Holds information about each example.
    """

    def __init__(self, directory, filename):
        """
        Constructor for an Instance object.

        Args:
            directory (str): Base directory where the example lives.
            filename (str): The name of the file of the example.
        """

        self.image = misc.imread( os.path.join(directory, 'image', filename) )
        self.image = self.image / 255.0
        self.mask  = misc.imread( os.path.join(directory, 'mask',  filename) )
        self.mask  = self.mask / 255.0

        # Parse filename to get parameters

        items = filename.split('_')

        # Represent orientation as sin/cos vector
        angle = np.deg2rad(float(items[0][-3:])-90)
        self.orientation = np.array([np.sin(angle), np.cos(angle)])

        self.identity = int(items[1])-1 # Identities are 1-indexed

        self.gender = np.array(getattr(Gender, items[3]))
        self.emotion = np.array(getattr(Emotion, items[4]))


    def create_identity_vector(self, identity_map):
        """
        Creates a one-in-k encoding of the instance's identity.

        Args:
            identity_map (dict): Mapping from identity to a unique index.
        """

        self.identity_vec = np.zeros(len(identity_map), dtype=np.float32)
        self.identity_vec[ identity_map[self.identity] ] = 1.


    def parameter_info(self):
        """
        Return length and shape information about the instance's parameters.

        Returns:
            dict, of parameter information.
        """

        info = dict()
        info['identity_len']    = len(self.identity_vec)
        info['gender_len']      = len(self.gender)
        info['orientation_len'] = len(self.orientation)
        info['emotion_len']     = len(self.emotion)
        info['image_shape']     = tuple(self.image.shape[0:2])
        return info


    def th_image(self):
        """
        Returns a Theano-ordered representation of the image.
        """

        image = np.empty((3,)+self.image.shape[0:2])
        for i in range(0, 3):
            image[i,:,:] = self.image[:,:,i]
        return image


    def th_mask(self):
        """
        Returns a Theano-ordered representation of the image.
        """

        mask = np.empty((1,)+self.mask.shape[0:2])
        mask[0,:,:] = self.mask[:,:,0]
        return mask


