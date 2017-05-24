#!/usr/bin/env sh
$CAFFE_ROOT/build/tools/lxf_get_features.bin   \
		$DATA_ROOT/webface/Partitions/test.txt \
		$DATA_ROOT \
		$DATA_ROOT/webface/webface_mean.binaryproto \
		$CAFFE_ROOT/deepal/webface_svm/deploy_test.prototxt  \
		$CAFFE_ROOT/deepal/webface_svm/deploy_test.prototxt  \
		$CAFFE_ROOT/deepal/webface_svm/snapshot/_current.caffemodel \
		$CAFFE_ROOT/deepal/webface_svm/test_features.txt \
                36380 \
		2>&1 | tee logs/get_test_features.log    
