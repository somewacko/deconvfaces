"""
facegen/model.py

Methods to build FaceGen models.

"""

from keras.layers import Convolution2D, Dense, LeakyReLU, Input, MaxPooling2D, \
        merge, Reshape, UpSampling2D
from keras.models import Model, model_from_yaml

from .instance import Emotion


def build_model(identity_len=57, orientation_len=2,
        emotion_len=Emotion.length(), initial_shape=(5,4), deconv_layers=5,
        num_kernels=[256, 256, 96, 96, 64, 64, 32], optimizer='adam'):
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

    # TODO: Parameter validation

    identity_input    = Input(shape=(identity_len,),    name='identity')
    orientation_input = Input(shape=(orientation_len,), name='orientation')
    emotion_input     = Input(shape=(emotion_len,),     name='emotion')

    # Hidden representation for input parameters

    id_fc = LeakyReLU()( Dense(512)(identity_input) )
    or_fc = LeakyReLU()( Dense(512)(orientation_input) )
    em_fc = LeakyReLU()( Dense(512)(emotion_input) )

    merged = merge([id_fc, or_fc, em_fc], mode='concat')

    params = LeakyReLU()( Dense(1024)(merged) )
    params = LeakyReLU()( Dense(1024)(params) )

    # Apply deconvolution layers

    height, width = initial_shape

    x = LeakyReLU()( Dense(height*width*num_kernels[0])(params) )
    x = Reshape((num_kernels[0], height, width))(x)

    for i in range(0, deconv_layers):
        # Pool and upsample
        x = MaxPooling2D((1,1))(x)
        x = UpSampling2D((2,2))(x)

        # Apply 5x5 and 3x3 convolutions

        # If we didn't specify the number of kernels to use for this many
        # layers, just repeat the last one in the list.
        idx = i if i < len(num_kernels) else -1
        x = LeakyReLU()( Convolution2D(num_kernels[idx], 5, 5, border_mode='same')(x) )
        x = LeakyReLU()( Convolution2D(num_kernels[idx], 3, 3, border_mode='same')(x) )

    # Last deconvolution layer: Create 3-channel image.
    x = MaxPooling2D((1,1))(x)
    x = UpSampling2D((2,2))(x)
    x = LeakyReLU()( Convolution2D(3, 5, 5, border_mode='same')(x) )
    x = Convolution2D(3, 3, 3, border_mode='same', activation='sigmoid')(x)

    # Compile the model

    model = Model(input=[identity_input, orientation_input,
            emotion_input], output=x)
    # TODO: Optimizer options
    model.compile(optimizer=optimizer, loss='mse')

    return model


def load_model(model_path, weights_path=''):
    """
    Loads a model from a given .yaml file.

    Args:
        model_path (str): Path to the model .yaml file.
    Args (optional):
        weights_path (str): Path to a weights file to load.
    Returns:
        keras.Model, the loaded model.
    """

    with open(model_path, 'r') as model_file:
        yaml_str = model_file.read()
        model = model_from_yaml(yaml_str)
        # TODO: Optimizer options
        model.compile(optimizer='adam', loss='mse')
        if weights_path:
            model.load_weights(weights_path)
    return model


