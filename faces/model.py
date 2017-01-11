"""
faces/model.py

Methods to build FaceGen models.

"""

from keras import backend as K
from keras.layers import BatchNormalization, Convolution2D, Dense, LeakyReLU, \
        Input, MaxPooling2D, merge, Reshape, UpSampling2D
from keras.models import Model

from .instance import Emotion, NUM_YALE_POSES


def build_model(identity_len=57, orientation_len=2, lighting_len=4,
                emotion_len=Emotion.length(), pose_len=NUM_YALE_POSES,
                initial_shape=(5,4), deconv_layers=5, num_kernels=None,
                optimizer='adam', use_yale=False, use_jaffe=False):
    """
    Builds a deconvolution FaceGen model.

    Args (optional):
        identity_len (int): Length of the identity input vector.
        orientation_len (int): Length of the orientation input vector.
        emotion_len (int): Length of the emotion input vector.
        initial_shape (tuple<int>): The starting shape of the deconv. network.
        deconv_layers (int): How many deconv. layers to use. More layers
            gives better resolution, although requires more GPU memory.
        num_kernels (list<int>): Number of convolution kernels for each layer.
        optimizer (str): The optimizer to use. Will only use default values.
    Returns:
        keras.Model, the constructed model.
    """

    print(initial_shape)

    if num_kernels is None:
        num_kernels = [128, 128, 96, 96, 32, 32, 16]

    # TODO: Parameter validation

    identity_input = Input(shape=(identity_len,), name='identity')

    if use_yale:
        lighting_input = Input(shape=(lighting_len,), name='lighting')
        pose_input = Input(shape=(pose_len,), name='pose')
    else:
        orientation_input = Input(shape=(orientation_len,), name='orientation')
        emotion_input = Input(shape=(emotion_len,), name='emotion')

    # Hidden representation for input parameters

    fc1 = LeakyReLU()( Dense(512)(identity_input) )
    fc2 = LeakyReLU()( Dense(512)(lighting_input if use_yale else orientation_input) )
    fc3 = LeakyReLU()( Dense(512)(pose_input if use_yale else emotion_input) )

    params = merge([fc1, fc2, fc3], mode='concat')
    params = LeakyReLU()( Dense(1024)(params) )

    # Apply deconvolution layers

    height, width = initial_shape

    print('height:', height, 'width:', width)

    x = LeakyReLU()( Dense(height*width*num_kernels[0])(params) )
    if K.image_dim_ordering() == 'th':
        x = Reshape((num_kernels[0], height, width))(x)
    else:
        x = Reshape((height, width, num_kernels[0]))(x)

    for i in range(0, deconv_layers):
        # Upsample input
        x = UpSampling2D((2,2))(x)

        # Apply 5x5 and 3x3 convolutions

        # If we didn't specify the number of kernels to use for this many
        # layers, just repeat the last one in the list.
        idx = i if i < len(num_kernels) else -1
        x = LeakyReLU()( Convolution2D(num_kernels[idx], 5, 5, border_mode='same')(x) )
        x = LeakyReLU()( Convolution2D(num_kernels[idx], 3, 3, border_mode='same')(x) )
        x = BatchNormalization()(x)

    # Last deconvolution layer: Create 3-channel image.
    x = MaxPooling2D((1,1))(x)
    x = UpSampling2D((2,2))(x)
    x = LeakyReLU()( Convolution2D(8, 5, 5, border_mode='same')(x) )
    x = LeakyReLU()( Convolution2D(8, 3, 3, border_mode='same')(x) )
    x = Convolution2D(1 if use_yale or use_jaffe else 3, 3, 3,
                      border_mode='same', activation='sigmoid')(x)

    # Compile the model

    if use_yale:
        model = Model(input=[identity_input, pose_input, lighting_input], output=x)
    else:
        model = Model(input=[identity_input, orientation_input, emotion_input], output=x)

    # TODO: Optimizer options
    model.compile(optimizer=optimizer, loss='msle')

    return model
