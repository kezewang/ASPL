cd Release
make clean
make -j8 all
cd ..

export ASPL_ROOT=/media/wangkeze/ASPL

export CAFFE_ROOT=$ASPL_ROOT/caffe-master
export SRC_ROOT=$ASPL_ROOT/src
export DATA_ROOT=$ASPL_ROOT/datasets

#Optional: Initialy finetune the CNN
#bash $CAFFE_ROOT/deepal/cacd_svm/finetune_init.sh
#bash $CAFFE_ROOT/deepal/webface_svm/finetune_init.sh

#cp $CAFFE_ROOT/deepal/cacd_svm/snapshot/_init_finetuned.caffemodel $CAFFE_ROOT/deepal/cacd_svm/snapshot/_current.caffemodel
#cp $CAFFE_ROOT/deepal/webface_svm/snapshot/_init_finetuned.caffemodel $CAFFE_ROOT/deepal/webface_svm/snapshot/_current.caffemodel

#bash $CAFFE_ROOT/examples/imagenet/init_cacd.sh
#bash $CAFFE_ROOT/examples/imagenet/init_webface.sh

$SRC_ROOT/Release/ASPL_honor 2>&1 | tee run.log
