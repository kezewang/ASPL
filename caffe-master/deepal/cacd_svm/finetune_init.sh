#!/usr/bin/env sh

#export ASPL_ROOT=/media/wangkeze/ASPL
#
#export CAFFE_ROOT=$ASPL_ROOT/caffe-master
#export SRC_ROOT=$ASPL_ROOT/src
#export DATA_ROOT=$ASPL_ROOT/datasets
bash $CAFFE_ROOT/examples/imagenet/create_cacd.sh
$CAFFE_ROOT/build/tools/caffe.bin train \
    --solver=$CAFFE_ROOT/deepal/cacd_svm/solver_init.prototxt -gpu 0 \
    -weights=$CAFFE_ROOT/deepal/bvlc_reference_caffenet.caffemodel \
	2>&1 | tee $CAFFE_ROOT/deepal/cacd_svm/logs/train.log        
cp $SRC_ROOT/cacd_aspl_iter_80000.caffemodel $CAFFE_ROOT/deepal/cacd_svm/snapshot/_init_finetuned.caffemodel 
