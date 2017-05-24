#!/usr/bin/env sh
export ASPL_ROOT=/media/wangkeze/ASPL

export CAFFE_ROOT=$ASPL_ROOT/caffe-master
export SRC_ROOT=$ASPL_ROOT/src
export DATA_ROOT=$ASPL_ROOT/datasets

bash $CAFFE_ROOT/examples/imagenet/create_finetune_webface.sh
$CAFFE_ROOT/build/tools/caffe.bin train \
    --solver=$CAFFE_ROOT/deepal/webface_svm/solver_aspl.prototxt -gpu 0 \
    -weights=$CAFFE_ROOT/deepal/webface_svm/snapshot/_current.caffemodel \
	2>&1 | tee $CAFFE_ROOT/deepal/webface_svm/logs/train.log        
cp $SRC_ROOT/webface_aspl_iter_80000.caffemodel $CAFFE_ROOT/deepal/webface_svm/snapshot/_current.caffemodel 
