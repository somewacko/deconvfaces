#!/usr/bin/env python3
"""
fg.py

CLI for training and interfacing with FaceGen models.

"""

import argparse
import sys
import types


# ---- Available commands

def train(args):
    """
    Command to train a FaceGen model.
    """

    parser = argparse.ArgumentParser(
            description = "Trains a FaceGen model using the Radboud Face Database",
            usage       = "fg <data> <output> [<args>]",
    )
    parser.add_argument('data', type=str, help=
            "Directory where RaFD data lives.")
    parser.add_argument('output', type=str, help=
            "Directory to output results to.")

    parser.add_argument('-b', '--batch-size', type=int, default=32, help=
            "Batch size to use while training.")
    parser.add_argument('-e', '--num-epochs', type=int, default=100, help=
            "The number of epochs to train.")

    parser.add_argument('-v', '--visualize', action='store_true', help=
            "Output intermediate results after each epoch.")

    args = parser.parse_args(sys.argv[2:])

    raise RuntimeError("Command 'train' is not implemented yet!")


def generate(args):
    """
    Command to generate faces with a FaceGen model.
    """

    parser = argparse.ArgumentParser(
            description = "Generate faces using a trained FaceGen model.",
            usage       = "fg [<args>]",
    )
    parser.add_argument('output', type=str, help=
            "Directory to output results to.")
    parser.add_argument('-m', '--model', type=str, required=True, help=
            "Model definition file to use.")
    parser.add_argument('-w', '--weights', type=str, required=True, help=
            "Weights file to load.")

    # TODO: How to specify what to generate?
    #
    # Should be able to say things like:
    #
    #   * Which parameters to sweep
    #   * Starting and destination parameters
    #   * Rate of changing parameters, linear or otherwise

    args = parser.parse_args(sys.argv[2:])

    raise RuntimeError("Command 'generate' is not implemented yet!")


# ---- Command-line invocation

if __name__ == '__main__':

    # Use all functions defined in this file as possible commands to run
    cmd_fns   = [x for x in locals().values() if isinstance(x, types.FunctionType)]
    cmd_names = sorted([fn.__name__ for fn in cmd_fns])
    cmd_dict  = {fn.__name__: fn for fn in cmd_fns}

    parser = argparse.ArgumentParser(
            description = "Generate faces using a deconvolution network.",
            usage       = "fg <command> [<args>]"
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

