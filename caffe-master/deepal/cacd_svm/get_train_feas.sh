#!/usr/bin/env sh
$CAFFE_ROOT/build/tools/lxf_get_features.bin   \
		$DATA_ROOT/cacd/train_rest.txt \
		$DATA_ROOT/cacd/cacd_train \
		$DATA_ROOT/cacd/cacd_mean.binaryproto \
		$CAFFE_ROOT/deepal/cacd_svm/deploy_rest.prototxt  \
		$CAFFE_ROOT/deepal/cacd_svm/deploy_rest.prototxt  \
		$CAFFE_ROOT/deepal/cacd_svm/snapshot/_current.caffemodel \
		$CAFFE_ROOT/deepal/cacd_svm/train_features.txt \
		40438 \
		2>&1 | tee logs/get_features.log    
