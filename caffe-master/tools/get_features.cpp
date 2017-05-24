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
#include "caffe/proto/caffe.pb.h"

using std::string;
using namespace caffe;

void writeData( std::ofstream& feaFile, const float* feas, const int fea_dim )
{
	int* sparseIndex = new int[fea_dim] ;
	float* sparseData = new float[fea_dim] ;
	int nonzeroNum = 0 ;

	for (int j = 0; j < fea_dim; ++j)
	{
		if ( abs( feas[j] ) > 0.000001 )
		{
			sparseIndex[nonzeroNum] = j ;
			sparseData[nonzeroNum] = feas[j] ;
			nonzeroNum++ ;
		}
	}

	feaFile.write( ( char* )&nonzeroNum, sizeof( int ) ) ;

	if ( 0 != nonzeroNum )
	{
		feaFile.write( (char*)sparseIndex, sizeof( int ) * nonzeroNum ) ;
		feaFile.write( (char*)sparseData, sizeof( float ) * nonzeroNum ) ;
	}

	delete [] sparseIndex ;
	delete [] sparseData ;
}

void readImg2Blob( float* input_data, const BlobProto& mean, const char* path )
{
	cv::Mat img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
  if ( NULL == img.data )
     std::cout << "image not found!" << path << std::endl ;
	cv::resize( img, img, cv::Size( 227, 227 ) ) ;
        
  int idx = 0 ;
	for (int c = 0; c < 3; ++c)
	{
		for (int h = 0; h < img.rows; ++h)
		{
			const cv::Vec3b* imgPtr = img.ptr<cv::Vec3b>( h ) ;
			for ( int w = 0; w < img.cols; ++w )
			{
				input_data[idx] = imgPtr[w][c] - mean.data( idx ) ;
				++idx ;
			}
		}
	}
}

int main(int argc, char** argv) {
	printf("argc in getfeatures.cpp: %d\n", argc);
	for (int i = 0; i <= 7; i++) {
		printf("%s\n", argv[i]);
	}
	const char* file_list = argv[1];
	const char* img_folder = argv[2];

	const char* mean_file = argv[3];
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

	const string prototxtBatch(argv[4]) ;
	const string prototxtSingle(argv[5]) ;
	const string model(argv[6]);

	Caffe::SetDevice(0);
	Caffe::set_mode(Caffe::GPU);

	std::ofstream feaFile ;
	feaFile.open( std::string(argv[7]).c_str(), std::ios::out | std::ios::binary );

	int smpNum = (int)samples.size() ;
	feaFile.write( ( char* )&smpNum, sizeof( int ) ) ;

	int64 e1 = cv::getTickCount();

//Batch Input
	Net<float> caffe_net(prototxtBatch, caffe::TEST);
	caffe_net.CopyTrainedLayersFrom(model);
	const vector<Blob<float>*>& input_blobs = caffe_net.input_blobs();
	const vector<Blob<float>* >& output_blobs1 = caffe_net.output_blobs();
	const int batchSize = output_blobs1[0]->num() ;
	std::cout << "Processing Batch image input: " << batchSize << std::endl ;

	for (size_t i = 0; i < samples.size() / batchSize; ++i)
	{
		float* input_data = input_blobs[0]->mutable_cpu_data();
		for ( int b = 0; b < batchSize; b++ )
		{
			char path[PATH_MAX];
			sprintf(path, "%s/%s", img_folder, samples[i * batchSize + b].first.c_str() );
			readImg2Blob( input_data + input_blobs[0]->offset( b ), mean, path ) ;
		}

		caffe_net.Forward();

		const vector<Blob<float>* >& output_blobs = caffe_net.output_blobs();
		const float* feas = output_blobs[0]->cpu_data();
		int fea_dim = output_blobs[0]->count() / batchSize ;
//        std::cout << output_blobs[0]->num() << " " << output_blobs[0]->channels() << " " << output_blobs[0]->height() << " " << output_blobs[0]->width() << std::endl ;

        /*if ( fea_dim != 4096 )
        {
            std::cout << "fea_dim" << fea_dim <<  " does not equal to 4096" << std::endl ;
            exit( 0 ) ;
        }*/

		for ( int b = 0; b < batchSize; b++ )
		{
			//std::cout << "feas " << output_blobs[0]->offset( b ) << std::endl;
			writeData( feaFile, feas + output_blobs[0]->offset( b ), fea_dim ) ;
		}
	}

//Single Input
	Net<float> caffe_netS(prototxtSingle, caffe::TEST);
	caffe_netS.CopyTrainedLayersFrom(model);
	const vector<Blob<float>*>& input_blobsS = caffe_netS.input_blobs();
	std::cout << "Processing Single image input" << std::endl ;
        std::cout << prototxtSingle << std::endl ;

	for (size_t i = samples.size() / batchSize * batchSize; i < samples.size(); ++i)
	{
		float* input_dataS = input_blobsS[0]->mutable_cpu_data();
		char path[PATH_MAX];
		sprintf(path, "%s/%s", img_folder, samples[i].first.c_str() );

		readImg2Blob( input_dataS, mean, path ) ;

		caffe_netS.Forward();

		const vector<Blob<float>* >& output_blobs = caffe_netS.output_blobs();
		const float* feas = output_blobs[0]->cpu_data();
		int fea_dim = output_blobs[0]->count() ;
		/*for (int t = 0; t < fea_dim; t++) {
			std::cout << feas[t] << " ";
		}
		std::cout << std::endl;*/
		writeData( feaFile, feas, fea_dim ) ;
	}


	feaFile.close() ;
	int64 e2 = cv::getTickCount();
	double time = (e2 - e1)/ cv::getTickFrequency();
	std::cout <<"feature gen time: "<<time<<std::endl;

	return 0;
}

