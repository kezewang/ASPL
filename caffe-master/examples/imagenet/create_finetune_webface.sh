#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

rm -r $CAFFE_ROOT/examples/imagenet/webface_finetune_lmdb/

EXAMPLE=$CAFFE_ROOT/examples/imagenet
DATA1=$CAFFE_ROOT/deepal/webface_svm/finetune.txt
TOOLS=$CAFFE_ROOT/build/tools

TRAIN_DATA_ROOT=$DATA_ROOT/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

echo "Creating finetune lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $TRAIN_DATA_ROOT \
    $DATA1 \
    $EXAMPLE/webface_finetune_lmdb

echo "Done."
