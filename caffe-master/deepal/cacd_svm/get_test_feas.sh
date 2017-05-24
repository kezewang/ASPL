#!/usr/bin/env sh
$CAFFE_ROOT/build/tools/lxf_get_features.bin   \
		$DATA_ROOT/cacd/test.txt \
		$DATA_ROOT/cacd/cacd_test \
		$DATA_ROOT/cacd/cacd_mean.binaryproto \
		$CAFFE_ROOT/deepal/cacd_svm/deploy_test.prototxt  \
		$CAFFE_ROOT/deepal/cacd_svm/deploy_test.prototxt  \
		$CAFFE_ROOT/deepal/cacd_svm/snapshot/_current.caffemodel\
		$CAFFE_ROOT/deepal/cacd_svm/test_features.txt     \
		11430	\
		2>&1 | tee logs/get_test_feas.log
