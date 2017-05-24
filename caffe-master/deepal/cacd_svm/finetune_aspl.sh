#!/usr/bin/env sh
bash $CAFFE_ROOT/examples/imagenet/create_finetune_cacd.sh
$CAFFE_ROOT/build/tools/caffe.bin train \
    --solver=$CAFFE_ROOT/deepal/cacd_svm/solver_aspl.prototxt -gpu 0 \
    -weights=$CAFFE_ROOT/deepal/cacd_svm/snapshot/_current.caffemodel \
	2>&1 | tee $CAFFE_ROOT/deepal/cacd_svm/logs/train.log        
cp $SRC_ROOT/cacd_aspl_iter_40000.caffemodel $CAFFE_ROOT/deepal/cacd_svm/snapshot/_current.caffemodel 
