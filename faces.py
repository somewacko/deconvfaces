#!/usr/bin/env python3
"""
fg

CLI for training and interfacing with face generating models.

"""

import argparse
import sys
import types


# ---- Available commands

def train():
    """
    Command to train a model.
    """

    parser = argparse.ArgumentParser(
            description = "Trains a model using the Radboud Face Database",
            usage       = "faces <data> [<args>]",
    )
    parser.add_argument('data', type=str, help=
            "Directory where RaFD data lives.")
    parser.add_argument('-o', '--output', type=str, default='output', help=
            "Directory to output results to.")

    parser.add_argument('-m', '--model', type=str, default='', help=
            "The model to load. If none specified, a new model will be made instead.")
    parser.add_argument('-b', '--batch-size', type=int, default=16, help=
            "Batch size to use while training.")
    parser.add_argument('-e', '--num-epochs', type=int, default=100, help=
            "The number of epochs to train.")
    parser.add_argument('-opt', '--optimizer', type=str, default='adam', help=
            "Optimizer to use, must be a valid optimizer included in Keras.")
    parser.add_argument('-d', '--deconv-layers', type=int, default=5, help=
            "The number of deconvolution layers to include in the model.")
    parser.add_argument('-k', '--kernels-per-layer', type=int, nargs='+', help=
            "The number of kernels to include in each layer.")

    parser.add_argument('-v', '--visualize', action='store_true', help=
            "Output intermediate results after each epoch.")
    parser.add_argument('--use-yalefaces', action='store_true', help=
            "Use YaleFaces data instead of RaFD")
    parser.add_argument('--use-jaffe', action='store_true', help=
            "Use JAFFE data instead of RaFD")

    args = parser.parse_args(sys.argv[2:])


    import faces.train

    if args.deconv_layers > 6:
        print("Warning: Having more than 6 deconv layers will create images "
              "larger than the original data! (and may not fit in memory)")

    faces.train.train_model(args.data, args.output, args.model,
        batch_size            = args.batch_size,
        num_epochs            = args.num_epochs,
        optimizer             = args.optimizer,
        deconv_layers         = args.deconv_layers,
        kernels_per_layer     = args.kernels_per_layer,
        generate_intermediate = args.visualize,
        use_yale              = args.use_yalefaces,
        use_jaffe             = args.use_jaffe,
        verbose               = True,
    )


def generate():
    """
    Command to generate faces with a trained model.
    """

    parser = argparse.ArgumentParser(
            description = "Generate faces using a trained model.",
            usage       = "faces [<args>]",
    )
    parser.add_argument('-m', '--model', type=str, required=True, help=
            "Model definition file to use.")
    parser.add_argument('-o', '--output', type=str, required=True, help=
            "Directory to output results to.")
    parser.add_argument('-f', '--gen-file', type=str, required=True, help=
            "YAML file that specifies the parameters to generate.")
    parser.add_argument('-b', '--batch_size', type=int, default=64, help=
            "Batch size to use while generating images.")
    parser.add_argument('-ext', '--extension', type=str, default='jpg', help=
            "Image file extension to use when saving images.")

    args = parser.parse_args(sys.argv[2:])

    import faces.generate

    faces.generate.generate_from_yaml(args.gen_file, args.model, args.output,
            batch_size=args.batch_size, extension=args.extension)


# ---- Command-line invocation

if __name__ == '__main__':

    # Use all functions defined in this file as possible commands to run
    cmd_fns   = [x for x in locals().values() if isinstance(x, types.FunctionType)]
    cmd_names = sorted([fn.__name__ for fn in cmd_fns])
    cmd_dict  = {fn.__name__: fn for fn in cmd_fns}

    parser = argparse.ArgumentParser(
            description = "Generate faces using a deconvolution network.",
            usage       = "faces <command> [<args>]"
    )
    parser.add_argument('command', type=str, help=
            "Command to run. Available commands: {}.".format(cmd_names))

    args = parser.parse_args([sys.argv[1]])

    cmd = None
    try:
        cmd = cmd_dict[args.command]
    except KeyError:
        sys.stderr.write('\033[91m')
        sys.stderr.write("\nInvalid command {}!\n\n".format(args.command))
        sys.stderr.write('\033[0m')
        sys.stderr.flush()

        parser.print_help()

    if cmd is not None:
        cmd()
