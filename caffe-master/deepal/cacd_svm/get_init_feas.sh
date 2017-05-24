#!/usr/bin/env sh

$CAFFE_ROOT/build/tools/lxf_get_features.bin   \
		$DATA_ROOT/cacd/train_init.txt \
		$DATA_ROOT/cacd/cacd_train \
		$DATA_ROOT/cacd/cacd_mean.binaryproto \
		$CAFFE_ROOT/deepal/cacd_svm/deploy_init.prototxt  \
		$CAFFE_ROOT/deepal/cacd_svm/deploy_init.prototxt  \
		$CAFFE_ROOT/deepal/cacd_svm/snapshot/_current.caffemodel \
		$CAFFE_ROOT/deepal/cacd_svm/init_features.txt \
		4237 \
		2>&1 | tee logs/get_features.log    
