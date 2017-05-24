#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <string>
#include <time.h>
#include <dirent.h>
#include "config.h"
#include "Solver.h"

using namespace std;
using namespace cv;


/*
 *   The main function for ASPL learning framework
 *   Please reference to the format of the given configure file
 *   train_id_path: the path to read the <path, label> for the training samples
	 train_fea_path: the path to save features of training samples
	 init_id_path: the path to read the <path, label> for the initialization samples
	 init_fea_path: the path to save features of initialization samples
	 test_id_path: the path to read the <path, label> for the testing samples
	 test_fea_path: the path to save features of testing samples
	 finetune_id_path: the path to save <path, label> for fine-tuning the CNN
	 finetune_script_path: the script path to fine-tune the CNN
	 get_features_script_path: the script path to extract the CNN features for training, initialization and testing samples
	 fea_dim: the dimension of feature vector
	 al_select_per_iter: the number of selected samples in the AL process
	 all_select_num: the number of selected samples in the AL + SPL
 */
int main(int argc, char** argv){

//	if(argc != 2){
//		cout <<"ASPL config_file_path"<<endl;
//		cout <<"program needs command line parameter which is the configure file path."<<endl;
//		exit(0);
//	}

	//Parse the configure file
	char argvPath[] = "config//config_cacd.txt" ;
//	char argvPath[] = "config//config_webface.txt" ;
	Config* config = new Config(argvPath);

    //Initialize the ASPL solver
	Solver* solver = new Solver(config);

	//Perform ASPL learning framework
	solver->solve();

	//Release all the resources
	delete config ;
	delete solver ;

	return 0;
}


