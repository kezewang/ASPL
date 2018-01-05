Active Self-Paced Learning for Cost-Effective and Progressive Face Identification (ASPL) from Keze Wang's research projects: http://hcp.sysu.edu.cn/active-self-paced-learning-for-cost-effective-and-progressive-face-identification/

-----------------------------------------------------------------------------------------------------------------------------
The source code is for educational and research use only without any warranty.
if you use any part of the source code, please cite related paper:
Liang Lin, Keze Wang, Deyu Meng, Wangmeng Zuo, and Lei Zhang, "Active Self-Paced Learning for Cost-Effective and Progressive Face Identification", IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), DOI: 10.1109/TPAMI.2017.2652459, 2017.

If you have any question about the code, please email kezewang@gmail.com

The code is tesed on Ubuntu 14.04 platform with OpenCV 3.1. 

Note that, the Support Vector Machine Classifier is from the liblinear library (https://www.csie.ntu.edu.tw/~cjlin/liblinear/), which is seamlessly included in the src folder.

Please reference the following steps to run the code.

0) Set up the caffe with OpenCV3.0 according to http://caffe.berkeleyvision.org/install_apt.html

1) Download the CACD dataset (http://bcsiriuschen.github.io/CARC/, you can download our verison through http://www.sysu-hcp.net/wp-content/uploads/2017/05/cacd.zip) and CASIA-WebFace dataset (http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html). The image data should be placed in ASPL/datasets/cacd and ASPL/datasets/webface

2) Modify the absolute path in ASPL/src/config/config_cacd.txt, ASPL/src/config/config_webface.txt and ASPL/src/make_run.sh to run the ASPL learning framework.
