#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/media/wangkeze/ASPL/caffe-master/examples/imagenet
DATA=/media/wangkeze/ASPL/datasets/webface/
TOOLS=/media/wangkeze/ASPL/caffe-master/build/tools

$TOOLS/compute_image_mean $EXAMPLE/webface_train_lmdb \
$DATA/webface_mean.binaryproto

#DATA=$DATA_ROOT/cacd/
#TOOLS=build/tools
#
#$TOOLS/compute_image_mean $EXAMPLE/cacd_train_lmdb \
#  $DATA/cacd_mean.binaryproto


echo "Done."
