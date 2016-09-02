#!/usr/bin/env sh
#
# Fit models with different optimizers to visualize training

batch_size=16
deconv_layers=5
num_epochs=200
output_dir=output.optimizers

./fg train RaFD --visualize \
    -d $deconv_layers \
    -b $batch_size \
    -e $num_epochs \
    -o $output_dir \
    --optimizer sgd

./fg train RaFD --visualize \
    -d $deconv_layers \
    -b $batch_size \
    -e $num_epochs \
    -o $output_dir \
    --optimizer adagrad

./fg train RaFD --visualize \
    -d $deconv_layers \
    -b $batch_size \
    -e $num_epochs \
    -o $output_dir \
    --optimizer adadelta

./fg train RaFD --visualize \
    -d $deconv_layers \
    -b $batch_size \
    -e $num_epochs \
    -o $output_dir \
    --optimizer adam

