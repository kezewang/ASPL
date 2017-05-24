#!/usr/bin/env sh

$CAFFE_ROOT/build/tools/lxf_get_features.bin   \
		$DATA_ROOT/webface/Partitions/train_init.txt \
		$DATA_ROOT \
		$DATA_ROOT/webface/webface_mean.binaryproto \
		$CAFFE_ROOT/deepal/webface_svm/deploy_init.prototxt  \
		$CAFFE_ROOT/deepal/webface_svm/deploy_init.prototxt  \
		$CAFFE_ROOT/deepal/webface_svm/snapshot/_current.caffemodel \
		$CAFFE_ROOT/deepal/webface_svm/init_features.txt \
                14984 \
		2>&1 | tee logs/get_features.log    
