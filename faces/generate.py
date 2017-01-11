"""
faces/generate.py

Methods for generating faces.

"""

import os
import yaml

import numpy as np
from scipy import interpolate
import scipy.misc
from tqdm import tqdm

from .instance import Emotion, NUM_YALE_POSES

class GenParser:
    """
    Class to parse and create inputs based on the parameters in a yaml file.
    """

    # Default parameters to use
    DefaultParams = {
        'mode'        : 'single',
        'constrained' : True,
        'id'          : None,
        'em'          : None,
        'or'          : None,
        'ps'          : None,
        'lt'          : None,
        'id_scale'    : 1.0,
        'id_step'     : 0.1,
        'id_min'      : None,
        'id_max'      : None,
        'em_scale'    : 1.0,
        'em_step'     : 0.1,
        'em_min'      : None,
        'em_max'      : None,
        'or_scale'    : 1.0,
        'or_step'     : 0.1,
        'or_min'      : None,
        'or_max'      : None,
        'ps_scale'    : 1.0,
        'ps_step'     : 0.1,
        'ps_min'      : None,
        'ps_max'      : None,
        'lt_scale'    : 1.0,
        'lt_step'     : 0.1,
        'lt_min'      : None,
        'lt_max'      : None,
        'num_images'  : '1s',
        'fps'         : 30,
        'keyframes'   : None,
    }

    def __init__(self, yaml_path):
        self.yaml_file = open(yaml_path, 'r')

        self.modes = {
            'single'     : self.mode_single,
            'random'     : self.mode_random,
            'drunk'      : self.mode_drunk,
            'interpolate': self.mode_interpolate,
        }

    def __del__(self):
        self.yaml_file.close()


    # Methods for generating inputs by mode

    def mode_single(self, params):
        """
        Generate network inputs for a single image.
        """

        if params['id'] is None:
            params['id'] = 0
        if params['em'] is None:
            params['em'] = 'neutral'
        if params['or'] is None:
            params['or'] = 0
        if params['ps'] is None:
            params['ps'] = 0
        if params['lt'] is None:
            params['lt'] = 0

        if params['dataset'] == 'YALE':
            inputs = {
                'identity': np.empty((1, params['num_ids'])),
                'pose': np.empty((1, NUM_YALE_POSES)),
                'lighting': np.empty((1, 4)),
            }
            inputs['identity'][0,:] = self.identity_vector(params['id'], params)
            inputs['pose'][0,:] = self.pose_vector(params['ps'], params)
            inputs['lighting'][0,:] = self.lighting_vector(params['lt'], params)
        else:
            inputs = {
                'identity': np.empty((1, params['num_ids'])),
                'emotion': np.empty((1, Emotion.length())),
                'orientation': np.empty((1, 2)),
            }

            inputs['identity'][0,:] = self.identity_vector(params['id'], params)
            inputs['emotion'][0,:] = self.emotion_vector(params['em'], params)
            inputs['orientation'][0,:] = self.orientation_vector(params['or'], params)

        return inputs


    def mode_random(self, params):
        """
        Generate random network inputs.
        """

        num_images = self.num_frames(params['num_images'], params)

        if params['dataset'] == 'YALE':
            inputs = {
                'identity': np.empty((num_images, params['num_ids'])),
                'pose': np.empty((num_images, NUM_YALE_POSES)),
                'lighting': np.empty((num_images, 4)),
            }
        else:
            inputs = {
                'identity':    np.empty((num_images, params['num_ids'])),
                'emotion':     np.empty((num_images, Emotion.length())),
                'orientation': np.empty((num_images, 2)),
            }

        for i in range(0, num_images):
            if params['id'] is None:
                inputs['identity'][i,:] = self.random_identity(params)
            else:
                inputs['identity'][i,:] = self.identity_vector(params['id'], params)

            if params['dataset'] == "YALE":
                if params['ps'] is None:
                    inputs['pose'][i,:] = self.random_pose(params)
                else:
                    inputs['pose'][i,:] = self.pose_vector(params['ps'], params)

                if params['lt'] is None:
                    inputs['lighting'][i,:], _ = self.random_lighting(params)
                else:
                    inputs['lighting'][i,:] = self.lighting_vector(params['lt'], params)
            else:
                if params['em'] is None:
                    inputs['emotion'][i,:] = self.random_emotion(params)
                else:
                    inputs['emotion'][i,:] = self.emotion_vector(params['em'], params)

                if params['or'] is None:
                    inputs['orientation'][i,:], _ = self.random_orientation(params)
                else:
                    inputs['orientation'][i,:] = self.orientation_vector(params['or'], params)

        return inputs


    def mode_drunk(self, params):
        """
        Generate "drunk" network inputs, random vectors created by randomly
        shifting the last vector.
        """

        num_images = self.num_frames(params['num_images'], params)

        if params['dataset'] == "YALE":
            inputs = {
                'identity': np.empty((num_images, params['num_ids'])),
                'pose':     np.empty((num_images, NUM_YALE_POSES)),
                'lighting': np.empty((num_images, 4)),
            }
        else:
            inputs = {
                'identity':    np.empty((num_images, params['num_ids'])),
                'emotion':     np.empty((num_images, Emotion.length())),
                'orientation': np.empty((num_images, 2)),
            }

        last_id, last_em, last_or, last_ps, last_lt = None, None, None, None, None

        for i in range(0, num_images):
            if params['id'] is None:
                inputs['identity'][i,:] = self.random_identity(params, last_id)
                last_id = inputs['identity'][i,:]
            else:
                inputs['identity'][i,:] = self.identity_vector(params['id'], params)

            if params['dataset'] == "YALE":
                if params['ps'] is None:
                    inputs['pose'][i,:] = self.random_pose(params, last_ps)
                    last_ps = inputs['pose'][i,:]
                else:
                    inputs['pose'][i,:] = self.pose_vector(params['ps'], params)

                if params['lt'] is None:
                    inputs['lighting'][i,:], last_lt = self.random_lighting(params, last_lt)
                else:
                    inputs['lighting'][i,:] = self.lighting_vector(params['lt'], params)
            else:
                if params['em'] is None:
                    inputs['emotion'][i,:] = self.random_emotion(params, last_em)
                    last_em = inputs['emotion'][i,:]
                else:
                    inputs['emotion'][i,:] = self.emotion_vector(params['em'], params)

                if params['or'] is None:
                    inputs['orientation'][i,:], last_or = self.random_orientation(params, last_or)
                else:
                    inputs['orientation'][i,:] = self.orientation_vector(params['or'], params)

        return inputs


    def mode_interpolate(self, params):
        """
        Generate network inputs that interpolate between keyframes.
        """

        use_yale = params['dataset'] == "YALE"

        # Set starting/default values
        id_val = params['id'] if params['id'] is not None else 0
        if use_yale:
            ps_val = params['ps'] if params['ps'] is not None else 0
            lt_val = params['lt'] if params['lt'] is not None else 0
        else:
            em_val = params['em'] if params['em'] is not None else 0
            or_val = params['or'] if params['or'] is not None else 0

        # List of all id/em/or vectors for each keyframe
        id_keyframes = list()
        if use_yale:
            ps_keyframes = list()
            lt_keyframes = list()
        else:
            em_keyframes = list()
            or_keyframes = list()

        keyframe_indicies = list()


        frame_index = None

        for keyframe_params in params['keyframes']:

            # Get new parameters, otherwise use values from the last keyframe
            if 'id' in keyframe_params: id_val = keyframe_params['id']
            if use_yale:
                if 'ps' in keyframe_params: ps_val = keyframe_params['ps']
                if 'lt' in keyframe_params: lt_val = keyframe_params['lt']
            else:
                if 'em' in keyframe_params: em_val = keyframe_params['em']
                if 'or' in keyframe_params: or_val = keyframe_params['or']

            # Determine which frame index this is in the animation
            if frame_index is None:
                frame_index = 0
            else:
                if 'length' not in keyframe_params:
                    raise RuntimeError("A length must be specified for every "
                                       "keyframe except the first")
                frame_index += self.num_frames(keyframe_params['length'], params)

            # Create input vectors for this keyframe
            id_keyframes.append( self.identity_vector(id_val, params) )
            if use_yale:
                ps_keyframes.append( self.pose_vector(ps_val, params) )
                lt_keyframes.append( self.lighting_vector(lt_val, params) )
            else:
                em_keyframes.append( self.emotion_vector(em_val, params) )
                or_keyframes.append( self.orientation_vector(or_val, params) )

            keyframe_indicies.append( frame_index )

        # Convert python lists to numpy arrays
        id_keyframes = np.vstack(id_keyframes)
        if use_yale:
            ps_keyframes = np.vstack(ps_keyframes)
            lt_keyframes = np.vstack(lt_keyframes)
        else:
            em_keyframes = np.vstack(em_keyframes)
            or_keyframes = np.vstack(or_keyframes)

        keyframe_indicies = np.array(keyframe_indicies)

        num_frames = keyframe_indicies[-1]+1

        # Interpolate
        if use_yale:
            id_idx = np.arange(0, NUM_YALE_ID)
            ps_idx = np.arange(0, NUM_YALE_POSES)
            lt_idx = np.arange(0, 4)
        else:
            id_idx = np.arange(0, NUM_ID)
            em_idx = np.arange(0, Emotion.length())
            or_idx = np.arange(0, 2)

        f_id = interpolate.interp2d(id_idx, keyframe_indicies, id_keyframes)
        if use_yale:
            f_ps = interpolate.interp2d(ps_idx, keyframe_indicies, ps_keyframes)
            f_lt = interpolate.interp2d(lt_idx, keyframe_indicies, lt_keyframes)
        else:
            f_em = interpolate.interp2d(em_idx, keyframe_indicies, em_keyframes)
            f_or = interpolate.interp2d(or_idx, keyframe_indicies, or_keyframes)

        if use_yale:
            return {
                'identity': f_id(id_idx, np.arange(0, num_frames)),
                'pose':     f_ps(ps_idx, np.arange(0, num_frames)),
                'lighting': f_lt(lt_idx, np.arange(0, num_frames)),
            }
        else:
            return {
                'identity':    f_id(id_idx, np.arange(0, num_frames)),
                'emotion':     f_em(em_idx, np.arange(0, num_frames)),
                'orientation': f_or(or_idx, np.arange(0, num_frames)),
            }


    # Helper methods

    def num_frames(self, val, params):
        """ Gets the number of frames for a value. """

        if isinstance(val, int):
            return val
        elif isinstance(val, str):
            if val.endswith('s'):
                return int( float(val[:-1]) * params['fps'] )
            else:
                raise RuntimeError("Length '{}' not understood".format(val))
        else:
            raise RuntimeError("Length '{}' not understood".format(val))


    def identity_vector(self, value, params):
        """ Create an identity vector for a provided value. """

        if isinstance(value, str):
            if '+' not in value:
                raise RuntimeError("Identity '{}' not understood".format(value))

            try:
                values = [int(x) for x in value.split('+')]
            except:
                raise RuntimeError("Identity '{}' not understood".format(value))
        elif isinstance(value, int):
            values = [value]
        else:
            raise RuntimeError("Identity '{}' not understood".format(value))

        vec = np.zeros((params['num_ids'],))
        for val in values:
            if val < 0 or params['num_ids'] <= val:
                raise RuntimeError("Identity '{}' invalid".format(val))
            vec[val] += 1.0

        return self.constrain(vec, params['constrained'], params['id_scale'],
                params['id_min'], params['id_max'])


    def emotion_vector(self, value, params):
        """ Create an emotion vector for a provided value. """

        if not isinstance(value, str):
            raise RuntimeError("Emotion '{}' not understood".format(value))

        if '+' in value:
            values = value.split('+')
        else:
            values = [value]

        vec = np.zeros((Emotion.length(),))
        for emotion in values:
            try:
                vec += getattr(Emotion, emotion)
            except AttributeError:
                raise RuntimeError("Emotion '{}' is invalid".format(emotion))

        return self.constrain(vec, params['constrained'], params['em_scale'],
                params['em_min'], params['em_max'])


    def orientation_vector(self, value, params):
        """ Create an orientation vector for a provided value. """

        if isinstance(value, int) or isinstance(value, float):
            value = np.deg2rad(value)
            return np.array([np.sin(value), np.cos(value)])

        elif isinstance(value, str):
            if params['constrained']:
                raise RuntimeError("Cannot manually set orientation vector "
                                   "values when constrained is set to True")

            values = value.split()
            if len(values) != 2:
                raise RuntimeError("Orientation '{}' not understood".format(value))

            vec = np.empty((2,))
            try:
                vec[0] = float(values[0])
                vec[1] = float(values[1])
            except ValueError:
                raise RuntimeError("Orientation '{}' not understood".format(value))

            return vec
        else:
            raise RuntimeError("Orientation '{}' not understood".format(value))


    def pose_vector(self, value, params):
        """ Create an pose vector for a provided value. """

        if isinstance(value, str):
            if '+' not in value:
                raise RuntimeError("Pose '{}' not understood".format(value))

            try:
                values = [int(x) for x in value.split('+')]
            except:
                raise RuntimeError("Pose '{}' not understood".format(value))
        elif isinstance(value, int):
            values = [value]
        else:
            raise RuntimeError("Pose '{}' not understood".format(value))

        vec = np.zeros((NUM_YALE_POSES,))
        for val in values:
            if val < 0 or NUM_YALE_POSES <= val:
                raise RuntimeError("Pose '{}' invalid".format(val))
            vec[val] += 1.0

        return self.constrain(vec, params['constrained'], params['ps_scale'],
                params['ps_min'], params['ps_max'])


    def lighting_vector(self, value, params):
        """ Create a lighting vector for a provided value. """

        if isinstance(value, int) or isinstance(value, float):
            value = np.deg2rad(value)
            return np.array([np.sin(value), np.cos(value), np.sin(value), np.cos(value)])

        elif isinstance(value, str):

            values = value.split()
            if len(values) != 2:
                raise RuntimeError("Lighting '{}' not understood".format(value))

            vec = np.empty((4,))
            try:
                # First element is azimuth
                vec[0] = np.sin(float(values[0]))
                vec[1] = np.cos(float(values[0]))
                # Second element is elevation
                vec[2] = np.sin(float(values[1]))
                vec[3] = np.cos(float(values[1]))
            except ValueError:
                raise RuntimeError("Lighting '{}' not understood".format(value))

            return vec
        else:
            raise RuntimeError("Lighting '{}' not understood".format(value))


    def random_identity(self, params, start=None):
        """ Create a random identity vector. """

        step = params['id_step']

        if start is None:
            vec = 2*(np.random.rand(params['num_ids'])-0.5)
        else:
            vec = start + (2*step*np.random.rand(params['num_ids'])-step)

        return self.constrain(vec, params['constrained'], params['id_scale'],
                params['id_min'], params['id_max'])


    def random_emotion(self, params, start=None):
        """ Create a random emotion vector. """

        step = params['em_step']

        if start is None:
            vec = 2*(np.random.rand(Emotion.length())-0.5)
        else:
            vec = start + (2*step*np.random.rand(Emotion.length())-step)

        return self.constrain(vec, params['constrained'], params['em_scale'],
                params['em_min'], params['em_max'])


    def random_orientation(self, params, start=None):
        """ Create a random orientation vector. """

        step = params['or_step']

        if params['constrained']:
            if start is None:
                angle = 180*np.random.rand() - 90
            else:
                angle = start + step * (180*np.random.rand()-90)
            rad = np.deg2rad(angle)

            # Return the angle as a second argument so the caller can grab it
            # in case it's in the drunk mode
            return np.array([np.sin(rad), np.cos(rad)]), angle
        else:
            if start is None:
                vec = 2*np.random.rand(2) - 1
            else:
                vec = start + (2*step*np.random.rand(2)-step)

            vec = self.constrain(vec, params['constrained'], params['or_scale'],
                    params['or_min'], params['or_max'])

            # Return the vector twice so it behaves the same as constrained
            return vec, vec


    def random_pose(self, params, start=None):
        """ Create a random pose vector. """

        step = params['ps_step']

        if start is None:
            vec = 2*(np.random.rand(NUM_YALE_POSES)-0.5)
        else:
            vec = start + (2*step*np.random.rand(NUM_YALE_POSES)-step)

        return self.constrain(vec, params['constrained'], params['ps_scale'],
                params['ps_min'], params['ps_max'])


    def random_lighting(self, params, start=None):
        """ Create a random lighting vector. """

        step = params['lt_step']

        if params['constrained']:
            if start is None:
                azimuth = 180*np.random.rand() - 90
                elevation = 180*np.random.rand() - 90
            else:
                azimuth = start[0] + step * (180*np.random.rand()-90)
                elevation = start[1] + step * (180*np.random.rand()-90)
            azrad = np.deg2rad(azimuth)
            elrad = np.deg2rad(elevation)

            # Return the angle as a second argument so the caller can grab it
            # in case it's in the drunk mode
            return np.array([np.sin(azrad), np.cos(azrad), np.sin(elrad),
                np.cos(elrad)]), (azimuth, elevation)
        else:
            if start is None:
                vec = 2*np.random.rand(4) - 1
            else:
                vec = start + (2*step*np.random.rand(4)-step)

            vec = self.constrain(vec, params['constrained'], params['lt_scale'],
                    params['lt_min'], params['lt_max'])

            # Return the vector twice so it behaves the same as constrained
            return vec, vec


    def constrain(self, vec, constrained, scale, vec_min, vec_max):
        """ Constrains the emotion vector based on params. """

        if constrained:
            vec = vec / np.linalg.norm(vec)

        if scale is not None:
            vec = vec * scale

        if vec_min is not None and vec_max is not None:
            vec = np.clip(vec, vec_min, vec_max)

        return vec


    # Main parsing method

    def parse_params(self):
        """
        Parses the yaml file and creates input vectors to use with the model.
        """

        self.yaml_file.seek(0)

        yaml_params = yaml.load(self.yaml_file)

        params = GenParser.DefaultParams

        for field in params.keys():
            if field in yaml_params:
                params[field] = yaml_params[field]

        return params

    def gen_inputs(self, params):
        """
        creates input vectors to use with the model.
        """

        fn = None
        try:
            fn = self.modes[ params['mode'] ]
        except KeyError:
            raise RuntimeError("Mode '{}' is invalid".format(params['mode']))

        return fn(params)



def generate_from_yaml(yaml_path, model_path, output_dir, batch_size=32,
        extension='jpg'):
    """
    Generate images based on parameters specified in a yaml file.
    """

    from keras import backend as K
    from keras.models import load_model

    print("Loading model...")

    model = load_model(model_path)
    num_ids = model.input_shape[0][1] # XXX is there a nicer way to get this?
    dataset = os.path.basename(model_path).split('.')[1]

    parser = GenParser(yaml_path)

    try:
        params = parser.parse_params()
    except RuntimeError as e:
        print("Error: Unable to parse '{}'. Encountered exception:".format(yaml_path))
        print(e)
        return
    params['dataset'] = dataset
    params['num_ids'] = num_ids
    inputs = parser.gen_inputs(params)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise RuntimeError(
            "Directory '{}' exists. Cowardly refusing to continue."
            .format(output_dir))

    print("Generating images...")

    num_images = inputs['identity'].shape[0]
    count = 0

    for idx in tqdm(range(0, num_images, batch_size)):

        if dataset == "YALE":
            batch = {
                'identity': inputs['identity'][idx:idx+batch_size,:],
                'pose':     inputs['pose']    [idx:idx+batch_size,:],
                'lighting': inputs['lighting'][idx:idx+batch_size,:],
            }
        else:
            batch = {
                'identity':    inputs['identity']   [idx:idx+batch_size,:],
                'emotion':     inputs['emotion']    [idx:idx+batch_size,:],
                'orientation': inputs['orientation'][idx:idx+batch_size,:],
            }

        gen = model.predict_on_batch(batch)

        for i in range(0, gen.shape[0]):
            if K.image_dim_ordering() == 'th':
                if dataset == "YALE":
                    image[:,:] = gen[i,0,:,:]
                else:
                    image = np.empty(gen.shape[2:]+(3,))
                    for x in range(0, 3):
                        image[:,:,x] = gen[i,x,:,:]
            else:
                if dataset == "YALE" or dataset == "JAFFE":
                    image = gen[i,:,:,0]
                else:
                    image = gen[i,:,:,:]
            image = np.array(255*np.clip(image,0,1), dtype=np.uint8)
            file_path = os.path.join(output_dir, '{:05}.{}'.format(count, extension))
            scipy.misc.imsave(file_path, image)
            count += 1
