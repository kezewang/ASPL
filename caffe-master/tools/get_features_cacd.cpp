#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <utility>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
//#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"

using std::string;
using namespace caffe;

int main(int argc, char** argv) {
	const char* file_list = argv[1];  //"/home/d302/liangdp/deepal/train.txt";
	const char* img_folder = argv[2]; //"/home/d302/liangdp/deepal/CACD_train";

	const char* mean_file = argv[3]; //"/home/d302/liangdp/deepal/cacd_mean.binaryproto";
	BlobProto mean;
	ReadProtoFromBinaryFileOrDie(mean_file, &mean);

	std::vector<std::pair<string, int> > samples;
	{
		std::pair<string, int> tmp_pair;
		std::ifstream input_file(file_list);
		while(input_file >> tmp_pair.first >> tmp_pair.second) {
			samples.push_back(tmp_pair);
		}
		input_file.close();
	}

	const string prototxt(argv[4]);
 			//"/home/d302/liangdp/playground/caffe-master/deepal/cacd/deploy.prototxt";
	const string model(argv[5]);
			//"/home/d302/liangdp/playground/caffe-master/deepal/snapshot/cacd/_iter_11000.caffemodel";

	Caffe::SetDevice(0);
	Caffe::set_mode(Caffe::GPU);
	Net<float> caffe_net(prototxt, caffe::TEST);
	caffe_net.CopyTrainedLayersFrom(model);

	const vector<Blob<float>*>& input_blobs = caffe_net.input_blobs();

	char path[PATH_MAX];
	cv::Mat img;
	float* input_data;
	int idx = 0;
	//"/home/d302/liangdp/playground/caffe-master/deepal/cacd/features.txt"
	std::ofstream feaFile(std::string(argv[6]).c_str());
	int fea_dim;

	 int64 e1;
	 int64 e2;
	 double time;
	 e1 = cv::getTickCount();

	 float entropy;
	 int max_index;
	 float max_pro;
	 float max_second;

	 float all_less = 0;
	 float correct_less = 0;

	 float accuracy = 0;

	for (size_t i = 0; i < samples.size(); ++i) {
		sprintf(path, "%s/%s", img_folder, samples[i].first.c_str());
		img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
		input_data = input_blobs[0]->mutable_cpu_data();

		idx = 0;
		for (int c = 0; c < 3; ++c) {
			for (int h = 0; h < img.rows; ++h) {
				for (int w = 0; w < img.cols; ++w, ++idx) {
					input_data[idx] = img.at<cv::Vec3b>(h, w)[c] - mean.data(idx);
				}
			}
		}
		caffe_net.ForwardPrefilled();

		entropy = 0;
		max_pro = -1;
		max_second = -1;

		const vector<Blob<float>* >& output_blobs = caffe_net.output_blobs();
		const float* feas = output_blobs[0]->cpu_data();
		fea_dim = output_blobs[0]->count();
		//std::cout << fea_dim << std::endl;
		for (int j = 0; j < fea_dim; ++j) {
			feaFile << feas[j] << " ";
		}

		feaFile << "\n";
		feaFile.flush();


	}
	e2 = cv::getTickCount();
	time = (e2 - e1)/ cv::getTickFrequency();

	//std::cout <<"accuracy: "<< accuracy / samples.size() <<std::endl;
	std::cout <<"test time: "<<time<<std::endl;

	return 0;
}

