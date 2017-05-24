/*
 * Solver.h
 *
 */

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <list>
#include <cv.h>
#include <ctime>
#include <map>
#include "Photo.hpp"
#include "config.h"
#include "Album.h"
#include "PersonClassifier.h"
#include "Logger.h"
using namespace std;
using namespace cv;

#ifndef SOLVER_H_
#define SOLVER_H_

typedef pair<int, float> Index_Lambda;
/*
 * Class Solver to train/test classifiers according to the album of photos
 */

class Solver{
public:
	Solver():thread_num( 4 ){
		this->config = NULL;
		this->album = NULL;
		this->iter = 0;
		//this->accuracy = 0;
	}
	Solver(Config* config);
	virtual ~Solver(){}
	void solve();
private:
	int queryLabel(Photo& photo){ return photo.getTrueLabel(); }
	void trainAllClassifiersOneVsALL() ;
	void testTrainedClassifiers();
	void trainAllClassifiersTogether( float C ) ;
	void testTrainedClassifiersTogether() ;
	void SPL() ;
	void updateLambda(const int iter) ;
	void AL() ;
	void verify() ;
        void initNewClassifierWithOneSmp( Index_Photo&, const int ) ;
	
	void fineTuneCNN(const int iter);
	void initialization();
	void reloadAlbum();
	void generateFineTuneIdFile(const char* filePath);
	void update_low_config_class();
	int get_iter();
	void print_spl_accuracy();
	void set_iter(const int iter);
	void AL_low_spl_selectFrequency();
private:
	Album* album;
	Config* config;
	map<int, PersonClassifier> classifiers_map;
	ofstream log_file;
	const int thread_num ;
	list<Index_Photo> unclear_set ;
	vector<int> low_config_class;
	vector<Index_Lambda> lambda;
	int iter;
	//double accuracy;
};

#endif /* SOLVER_H_ */
