#!/usr/bin/env sh

$CAFFE_ROOT/build/tools/lxf_get_features.bin   \
		$DATA_ROOT/webface/Partitions/train_rest.txt \
		$DATA_ROOT \
		$DATA_ROOT/webface/webface_mean.binaryproto \
		$CAFFE_ROOT/deepal/webface_svm/deploy_rest.prototxt  \
		$CAFFE_ROOT/deepal/webface_svm/deploy_rest.prototxt  \
		$CAFFE_ROOT/deepal/webface_svm/snapshot/_current.caffemodel \
		$CAFFE_ROOT/deepal/webface_svm/train_features.txt \
                130537 \
		2>&1 | tee logs/get_train_features.log    
