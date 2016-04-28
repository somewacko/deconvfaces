#!/usr/bin/env python3
"""
Learn a deconvolution network to generate and interpolate between faces.

Adapted from A. Dosovitskiy et al, "Learning to Generate Chairs, Tables, and
Cars with Convolutional Networks" 2014.

"""

import argparse
import os
import sys

from keras.callbacks import Callback
from keras.layers import Convolution2D, Dense, LeakyReLU, Input, MaxPooling2D, \
        merge, Reshape, UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras.utils.layer_utils import print_summary

import numpy as np
import scipy.misc as misc

from instance import Emotion, Instance, load_data



def build_model(identity_len=57, gender_len=2, orientation_len=2,
        emotion_len=8, output_shape=(160,128)):
    """
    Builds the deconvolution model.
    """

    print("Building model...")

    identity_input    = Input(shape=(identity_len,),    name='identity')
    gender_input      = Input(shape=(gender_len,),      name='gender')
    orientation_input = Input(shape=(orientation_len,), name='orientation')
    emotion_input     = Input(shape=(emotion_len,),     name='emotion')

    # Hidden representation for input parameters

    id_fc = LeakyReLU()( Dense(512)(identity_input) )
    gd_fc = LeakyReLU()( Dense(512)(gender_input) )
    or_fc = LeakyReLU()( Dense(512)(orientation_input) )
    em_fc = LeakyReLU()( Dense(512)(emotion_input) )

    merged = merge([id_fc, gd_fc, or_fc, em_fc], mode='concat')

    params = LeakyReLU()( Dense(1024)(merged) )
    params = LeakyReLU()( Dense(1024)(params) )

    # Get dimensions of initial volume

    height = int(output_shape[0]/16)
    width  = int(output_shape[1]/16)

    # Create RGB stream

    x = LeakyReLU()( Dense(height*width*256)(params) )
    x = Reshape((256, height, width))(x)

    x = MaxPooling2D((1,1))(x)
    x = UpSampling2D((2,2))(x)
    x = LeakyReLU()( Convolution2D(256, 5, 5, border_mode='same')(x) )

    x = MaxPooling2D((1,1))(x)
    x = UpSampling2D((2,2))(x)
    x = LeakyReLU()( Convolution2D(92, 5, 5, border_mode='same')(x) )

    x = MaxPooling2D((1,1))(x)
    x = UpSampling2D((2,2))(x)
    x = LeakyReLU()( Convolution2D(92, 5, 5, border_mode='same')(x) )

    x = MaxPooling2D((1,1))(x)
    x = UpSampling2D((2,2))(x)
    x = Convolution2D(3, 5, 5, border_mode='same', activation='sigmoid')(x)

    # TODO: Also create segmentation stream

    model = Model(input=[identity_input, gender_input, orientation_input,
            emotion_input], output=x)
    model.compile(optimizer='adam', loss='mse')

    return model


class GenerateIntermediate(Callback):
    """ Callback to generate intermediate images after each epoch. """

    def __init__(self, output_dir, parameters):
        """
        Constructor for a GenerateIntermediate object.

        Args:
            output_dir (str): Directory to save intermediate results in.
            parameters (dict): Parameters to input to the model.
        """
        super(Callback, self).__init__()

        self.output_dir = output_dir
        self.parameters = parameters

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


    def on_epoch_end(self, epoch, logs={}):
        """ Generate and save results to """

        dest_dir = os.path.join(self.output_dir, 'gen', 'e{:03}'.format(epoch))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        print("Generating images to {}".format(dest_dir))

        gen = self.model.predict(self.parameters)

        for i in range(0, gen.shape[0]):
            image = np.empty(gen.shape[2:]+(3,))
            for x in range(0, 3):
                image[:,:,x] = gen[i,x,:,:]
            image = np.array(255*np.clip(image,0,1), dtype=np.uint8)
            file_path = os.path.join(dest_dir, '{:03}.jpg'.format(i))
            misc.imsave(file_path, image)


# ---- Commands

def train(data_dir, output_dir, batch_size=128, num_epochs=100):
    """
    Trains the model on the data, generating intermediate results every epoch.

    Args:
        data_dir (str): Directory where the data lives.
        output_dir (str): Directory where outputs should be saved.
    Args (optional):
        batch_size (int): Size of the batch to use.
        num_epochs (int): Number of epochs to train for.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    instances = load_data(data_dir)

    # Build the model

    info = instances[0].parameter_info()

    model = build_model(
            identity_len    = info['identity_len'],
            gender_len      = info['gender_len'],
            orientation_len = info['orientation_len'],
            emotion_len     = info['emotion_len'],
            output_shape    = info['image_shape'],
    )

    # Load data into input tensors

    id_feat = np.empty((len(instances), info['identity_len']))
    gd_feat = np.empty((len(instances), info['gender_len']))
    or_feat = np.empty((len(instances), info['orientation_len']))
    em_feat = np.empty((len(instances), info['emotion_len']))

    images = np.empty((len(instances), 3)+info['image_shape'])

    for idx, instance in enumerate(instances):
        id_feat[idx,:] = instance.identity_vec
        gd_feat[idx,:] = instance.gender
        or_feat[idx,:] = instance.orientation
        em_feat[idx,:] = instance.emotion
        images[idx,:,:,:] = instance.th_image()

    model_inputs = {
        'identity'    : id_feat,
        'gender'      : gd_feat,
        'orientation' : or_feat,
        'emotion'     : em_feat,
    }

    # Create parameters to generate

    id_gen = np.zeros((54, info['identity_len']))
    gd_gen = np.zeros((54, info['gender_len']))
    or_gen = np.zeros((54, info['orientation_len']))
    em_gen = np.zeros((54, info['emotion_len']))

    # TODO: More dynamic way to do this
    x = 0
    for identity in range(0, 10+1, 5): # 3
        for gender in range(0, 2): # 2
            for angle in [-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90]: # 9
                angle = np.deg2rad(angle)

                id_gen[x,identity] = 1.
                gd_gen[x,gender] = 1.
                or_gen[x,:] = np.array([np.sin(angle), np.cos(angle)])
                em_gen[x,:] = Emotion.neutral

                x += 1

    # Give to callback

    gen_inputs = {
        'identity'    : id_gen,
        'gender'      : gd_gen,
        'orientation' : or_gen,
        'emotion'     : em_gen,
    }

    gen_intermediate = GenerateIntermediate(output_dir, gen_inputs)

    # Begin training

    model.fit(model_inputs, images, nb_epoch=num_epochs, batch_size=128,
            callbacks=[gen_intermediate])

    print("Saving model and weights...")

    yaml_str = model.to_yaml()
    f = open(os.path.join(output_dir, 'model.yaml'), 'w')
    f.write(yaml_str)
    f.close()

    model.save_weights(os.path.join(output_dir, 'weights.h5'), overwrite=True)

 
def generate():
    pass


# ---- Command-line invocation

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
            "Learn a deconvolution network to generate faces.")
    parser.add_argument('command', type=str, help=
            "Command to run. Either 'train', or 'generate'.")

    parser.add_argument('-d', '--data', type=str, default='processed/', help=
            "Directory where the data lives.")
    parser.add_argument('-o', '--output', type=str, default='output/', help=
            "Directory to place output images.")
    parser.add_argument('-m', '--model', type=str, default='model.yaml', help=
            "Model file to load/save.")
    parser.add_argument('-w', '--weights', type=str, default='', help=
            "Weights file to load/save.")

    parser.add_argument('-b', '--batch-size', type=int, default=128, help=
            "Size of the batch to use.")
    parser.add_argument('-e', '--num-epochs', type=int, default=100, help=
            "The number of epochs to train on.")

    parser.add_argument('--include-children', action='store_true', help=
            "Include children in the dataset. (Not implemented)")
    parser.add_argument('--use-gaze', action='store_true', help=
            "Use all gazes of each subject. (Not implemented)")

    args = parser.parse_args()

    if args.command == 'train':
        train(args.data, args.output, batch_size=args.batch_size,
                num_epochs=args.num_epochs)
    elif args.command == 'generate':
        generate()


