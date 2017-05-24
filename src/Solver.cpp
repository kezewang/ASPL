#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cv.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <sstream>
#include <cmath>
#include "Solver.h"
#include "Utils.h"
#include "train.h"

using namespace std;
using namespace cv;

#define LOG Logger(log_file, cout)

const string end = "\n";

void run_script(const string& script_path){
	FILE *fp = popen(script_path.c_str(), "r");
	if(fp == NULL) return;
	const int BUFF_L = 10240;
	char line[BUFF_L];
//	string *fake;
	while(fgets(line, BUFF_L, fp) != NULL){
		//cout << line ;
		//fake = new string(line);
		//if(fake->find("Iteration") != string::npos && fake->find("lr =") == string::npos && fake->find("loss =") == string::npos )
		//	cout << line;
		//if( fake->find("accuracy =") != string::npos )
		//	cout << line ;
		//		if( fake->find("Batch") != string::npos )
		//			cout << line ;
	}
	pclose(fp);
}

Solver::Solver(Config* config):thread_num( omp_get_max_threads() * 3 / 4 ){
	this->config = config;

	//Extract features
	cout << "Extracting initial features for training and testing samples..." << endl ;
	run_script(this->config->get_features_script_path()) ;
	cout << "Extracting initial features for training and testing samples done!" << endl ;

	this->album = new Album() ;

	this->album->loadPhotos(this->config) ;

	cout << "Training Sample Number: " << this->album->database.size()<<", Category Number: "<<this->album->validation_set.size()<<endl;

	this->log_file.open("log.txt");
	this->iter = 0;
}

void Solver::updateLambda(const int iter)
{
	cout << "start update lambda ..." << endl;
	for(map<int, PersonClassifier>::iterator it=this->classifiers_map.begin(); it!=this->classifiers_map.end(); ++it)
	{
		if ( it->second.isTrained() )
			it->second.update_spl_dis_threshold() ;

		//cout << "Lambda for" << it->first << "th classifier is " << it->second.get_spl_dis_threshold() << endl ;
	}
//	this->config->update_all_select_num(iter);
}



void Solver::solve(){
	int64 e1;
	int64 e2;
	double time;

	e1 = cv::getTickCount();

	this->initialization();

	e2 = cv::getTickCount();
	time = (e2 - e1)/ cv::getTickFrequency();
	LOG << "Initialization time: " << time << end << end;

	double all_time = 0;
	for(int i=0; i < 53; i++)
	{
		e1 = cv::getTickCount();
		LOG << "----------- iteration "<< i <<"----------" << end;
		this->iter = i;

		this->trainAllClassifiersOneVsALL() ;
		this->testTrainedClassifiers();

		this->verify() ;

		if ( (i+1) % 5 == 0)
		{
			this->AL();
			this->updateLambda(i);
		}

		this->SPL();
		this->fineTuneCNN(i);

		e2 = cv::getTickCount();
		time = (e2 - e1)/ cv::getTickFrequency();
		LOG << "iteration " << i << " run time: " << time << end << end;
		all_time += time;
	}

        this->trainAllClassifiersOneVsALL() ;
	this->testTrainedClassifiers() ;
	LOG << "total run time: " << all_time << end << end;
}

void Solver::initialization(){
	// load origin id and feature, and train all classifier

	vector<int> selected_index;
	this->album->getSelectedIndex(selected_index);
	int size = selected_index.size();

	for(int i=0; i<size; ++i)
	{
		int queryLabel = this->queryLabel(this->album->database[selected_index[i]].second);
		this->album->database[selected_index[i]].second.isQueryLabel = true;
		map<int, PersonClassifier>::iterator it = this->classifiers_map.find(queryLabel);
		if(it == this->classifiers_map.end()){
			PersonClassifier* classifier = new PersonClassifier(this->config);
			classifier->setPersonLabel(queryLabel);
			classifier->addAlPositives(this->album->database[selected_index[i]]);
			this->classifiers_map.insert(pair<int, PersonClassifier>(queryLabel, *classifier));
		}else{
			it->second.addAlPositives(this->album->database[selected_index[i]]);
		}
	}
}

void Solver::fineTuneCNN(const int iter){
	LOG << "fine-tuning cnn ..." << end;

	cout << "Generate (image, label) for fine-tuning..." << endl ;
	this->generateFineTuneIdFile(this->config->get_finetune_id_path().c_str());
	cout << "Generate (image, label) for fine-tuning done!" << endl << endl ;

	cout << "Create LMDB and fine-tune features..." << endl ;
	run_script(this->config->get_finetune_script_path());
	/*if (true)

	else
		run_script("/media/wangkeze/CEAL/caffe-master/deepal/cacd_svm/finetune_aspl2.sh");*/
	cout << "Create LMDB and fine-tune features done" << endl ;

	cout << "Extracting features for training and testing samples..." << endl ;
	run_script(this->config->get_features_script_path());
	cout << "Extracting features for training and testing samples done!" << endl ;

	this->reloadAlbum();

	for ( map<int, PersonClassifier>::iterator it = this->classifiers_map.begin(); it != this->classifiers_map.end(); it++ )
		it->second.setTrained( false ) ;

	LOG << "fine-tuning cnn done." << end << end;
}

void Solver::reloadAlbum(){
	//cout << "reloading album ..." <<endl;
	this->album->set_iter(this->iter);
	this->album->reLoadPhotos(this->config);

	for(map<int, PersonClassifier>::iterator it=this->classifiers_map.begin(); it!=this->classifiers_map.end(); ++it)
	{
		vector<int> po_index;
		it->second.getAlPositivesDatabaseIndex(po_index);
		vector<Index_Photo> new_al_pos;
		for(int i=0; i< (int)po_index.size(); ++i){
			new_al_pos.push_back(this->album->database[po_index[i]]);
		}

		vector<AL_Wrong_Index> new_al_wrong;
		it->second.get_AL_incorrect(new_al_wrong);

		it->second.setAlPositives(new_al_pos, new_al_wrong);
	}

	this->album->reLoadValidationSet(this->config);

	//cout << "reload album done." <<endl<<endl;
}

void Solver::generateFineTuneIdFile(const char* filePath)
{
	vector<int> selected_index;
	this->album->getSelectedIndex(selected_index);

	map<int, int> tmpLabels ;
	int newLabelId = 0 ;
	for ( map<int, PersonClassifier>::iterator it = this->classifiers_map.begin(); it != this->classifiers_map.end(); it++ )
	{
		if ( it->second.isTrained() )
		{
			tmpLabels.insert( pair<int, int>( it->second.getPersonLabel(), newLabelId ) ) ;
			newLabelId++ ;
		}
	}

	//Write file for fine-tuning
	vector<string> finetune_ids;
	for(size_t i=0; i < selected_index.size(); ++i)
	{
		ostringstream temp;
		int assignedLabel = this->album->database[selected_index[i]].second.getAssignedLabel() ;
		map<int, int>::const_iterator it = tmpLabels.find( assignedLabel ) ;
		if ( it != tmpLabels.end() )
		{
			temp << this->album->database[selected_index[i]].second.getName() <<" "<< tmpLabels[assignedLabel] ;
			finetune_ids.push_back(temp.str());
		}
	}
	cout << "The finetune file is: " << filePath << endl;
	cout << "The number of finetune samples is: " << finetune_ids.size() << endl;
	ofstream finetuneFile;
	finetuneFile.open(filePath);
	std::random_shuffle(finetune_ids.begin(), finetune_ids.end());
	for(size_t i=0; i < finetune_ids.size(); ++i){
		finetuneFile << finetune_ids[i] <<endl;
		finetuneFile.flush();
	}
	finetuneFile.close();

	// update train rest files list
	/*vector<int> not_selected;
	this->album->getNotSelectedIndex(not_selected);
	vector<string> train_rest_ids;
	for (size_t i = 0; i < not_selected.size(); ++i) {
		int groundTruthLabel = this->album->database[not_selected[i]].second.getTrueLabel();
		map<int, int>::const_iterator it = tmpLabels.find(groundTruthLabel);
		if (it != tmpLabels.end()) {
			ostringstream tmp;
			tmp << this->album->database[not_selected[i]].second.getName() << " " << tmpLabels[groundTruthLabel];
			train_rest_ids.push_back(tmp.str());
		}
	}
	string train_rest_file_path("/media/wangkeze/CEAL/caffe-master/deepal/cacd_svm/finetune_rest.txt");
	cout << "The train rest file path is: " << train_rest_file_path << endl;
	cout << "The number of train rest samples is: " << train_rest_ids.size() << endl;
	ofstream train_rest_file;
	train_rest_file.open(train_rest_file_path.c_str());
	for (size_t i = 0; i < train_rest_ids.size(); ++i) {
		train_rest_file << train_rest_ids[i] << endl;
		train_rest_file.flush();
	}
	train_rest_file.close();*/

	//update temporary labels for testing
	/*vector<string> test_ids;
	for(map<int, vector<Photo> >::iterator vali_it=this->album->validation_set.begin(); vali_it != this->album->validation_set.end(); ++ vali_it)
	{
		int groundTruthLabel = vali_it->first ;
		map<int, int>::const_iterator it = tmpLabels.find( groundTruthLabel ) ;
		if ( it != tmpLabels.end() )
		{
			for ( size_t i = 0; i < vali_it->second.size(); i++ )
			{
				ostringstream temp;
				temp << vali_it->second[i].getName() << " " << tmpLabels[groundTruthLabel] ;
				test_ids.push_back( temp.str() ) ;
			}
		}
	}

	string testfilePath( filePath ) ;
	testfilePath = testfilePath.substr( 0, testfilePath.find_last_of( ".txt" ) - 3 ) ;
	testfilePath += "_validation.txt" ;

	cout << "The test file path is: "<< testfilePath << endl ;
	cout << "The number of test samples is: " << test_ids.size() << endl;

	ofstream testFile ;
	testFile.open( testfilePath.c_str() ) ;
	for ( size_t i = 0; i < test_ids.size(); i++ )
	{
		testFile << test_ids[i] << endl ;
	    testFile.flush() ;
	}
	testFile.close() ;*/

	cout << "Current Classifier Number is " << newLabelId <<  " ! " << endl ;
	cout << "Please modify the classifier layer (i.e., fc8) inside train_val_aspl.prototxt" << endl ;
	cout << "Press any key to continue!" ;
//	std::cin.get();
}

bool contains(vector<int>& vec, int elem){
	bool flag = false;
	int size = vec.size();
	for(int i=0; i<size; ++i)
		if(vec[i] == elem){
			flag = true;
			break;
		}
	return flag;
}


//自定义排序函数
bool ascending(std::pair<float, int> i, std::pair<float, int> j) { return (i.first < j.first); }
bool descending(std::pair<float, int> i, std::pair<float, int> j) { return (i.first > j.first); }


Mat getFea(vector<Index_Photo> photos, int fea_dim){
	Mat result = Mat::zeros( photos.size(), fea_dim, CV_32F);

	for(size_t i=0; i<photos.size(); ++i)
	{
		SparseMat_<float> sparseDat ;
		photos[i].second.getSparseFeature().copyTo( sparseDat ) ;

		float* re_ptr = result.ptr<float>(i);
		for (SparseMatIterator_<float> it = sparseDat.begin(); it != sparseDat.end(); it++ )
		{
			const SparseMat::Node *n = it.node() ;
			re_ptr[n->idx[0]] = it.value<float>()  ;
		}
	}

	return result;
}

void Solver::SPL()
{
	vector<int> not_selected ;
	this->album->getNotSelectedIndex(not_selected) ;

	vector<int> satisfy_classifier_label;
	for(map<int, PersonClassifier>::iterator it=this->classifiers_map.begin(); it!=this->classifiers_map.end(); ++it)
	{
		if(it->second.isTrained())
		{
			it->second.loadXMl() ;
			satisfy_classifier_label.push_back(it->first);
		}
		it->second.current_al_po_size = it->second.getAlPositiveNum();
		vector<int> spl_index ;
		it->second.getSplPositivesDatabaseIndex( spl_index ) ;
		it->second.clearSplPositives() ;
		for ( size_t i = 0; i < spl_index.size(); i++ )
		{
			this->album->setNotSelected( spl_index[i] ) ;
			this->album->database[spl_index[i]].second.setAssignLabel( LABEL_NOT_ASSIGNED ) ;
		}
	}

	int al_random_select_num = min( (int)not_selected.size(), this->config->get_all_select_num() ) ;
	if ( 0 == al_random_select_num )
		return ;

	cout << endl << "Start SPL ..." << endl ;
	int count = 0 ;
	//float accuracy = 0;
	struct feature_node *x = (struct feature_node *) malloc(5000*sizeof(struct feature_node));
	for ( int i = 0; i < al_random_select_num; i++ )
	{
		SparseMat_<float> sparseData ;
		this->album->database[not_selected[i]].second.getSparseFeature().copyTo( sparseData )  ;
		int offset = 0 ;
		for (cv::SparseMatIterator_<float> it = sparseData.begin(); it != sparseData.end(); it++ )
		{
			const SparseMat::Node *n = it.node() ;
			x[offset].index = n->idx[0] + 1 ;
			x[offset].value = it.value<float>() ;
			++offset ;
		}

		vector<float> dis_list( satisfy_classifier_label.size() ) ;
#pragma omp parallel for num_threads( thread_num ) 
		for(size_t j=0; j<satisfy_classifier_label.size(); ++j)
		{
			map<int, PersonClassifier>::iterator it = this->classifiers_map.find(satisfy_classifier_label[j]);
			float temp_dis = it->second.getDis( x, offset );
			//sum_dis += abs( max( -1.f, min( temp_dis, 1.f ) ) + 1 ) ;
			dis_list[j] = ( 1 - temp_dis < it->second.get_spl_dis_threshold() ? 1 : -1) ;
		}

		float sum_dis = 0 ;
		float max_dis = -1 ;
		int predicted_label = -1 ;
		for(size_t j=0; j<satisfy_classifier_label.size(); ++j)
		{
			sum_dis += 1 + dis_list[j] ;

			if ( max_dis < dis_list[j] )
			{
				max_dis = dis_list[j] ;
				predicted_label = satisfy_classifier_label[j] ;
			}
		}

		if ( sum_dis == 2 )
		{
			map<int, PersonClassifier>::iterator it = this->classifiers_map.find( predicted_label );
			it->second.addSplPositives( this->album->database[not_selected[i]], min( 1.f, 1 - ( 1 - max_dis ) / it->second.get_spl_dis_threshold() ) ) ;
			this->album->setSelected( this->album->database[not_selected[i]].first ) ;
			this->album->database[not_selected[i]].second.setAssignLabel( predicted_label ) ;
			count++ ;
		}
		else if ( 0 == sum_dis )
		{
            if ( !this->album->isUnclear( this->album->database[not_selected[i]].first ) )
            {
            	this->unclear_set.push_back( this->album->database[not_selected[i]] ) ;
            	this->album->setUnclear( this->album->database[not_selected[i]].first ) ;
            }
		}
	}

	for(size_t j = 0; j < satisfy_classifier_label.size(); ++j)
	{
		map<int, PersonClassifier>::iterator it = this->classifiers_map.find(satisfy_classifier_label[j]);
		it->second.deleteSVM() ;
	}
	free( x ) ;

	cout << "SPL Selected Number: " << count << endl ;
	cout << "SPL done." <<endl << endl ;
}

void Solver::AL()
{
	cout << "Start AL ..." <<endl;

	vector<PersonClassifier> currentClassifiers ;
	vector<int> satisfy_classifier_label;
	for(map<int, PersonClassifier>::iterator it=this->classifiers_map.begin(); it!=this->classifiers_map.end(); ++it)
	{
		if(it->second.isTrained()){
			satisfy_classifier_label.push_back(it->first);
			it->second.loadXMl() ;
			it->second.current_al_po_size = it->second.getAlPositiveNum();
			currentClassifiers.push_back( it->second ) ;
		}
	}

	vector<int> not_selected ;
	this->album->getNotSelectedIndex(not_selected, 1) ;
	vector<int> random_index ;
	//getRandomList(not_selected.size(), random_index) ;
	for (size_t i = 0; i < not_selected.size(); ++i)
		random_index.push_back(i);

	const int al_random_select_num = min( (int)not_selected.size(), this->config->get_all_select_num() ) ;
	if ( 0 == al_random_select_num )
	{
		return ;
	}

	vector<pair<float, int> > index_dis;
	for(int i = 0; i < al_random_select_num; ++i)
	{
		float max_dis = -1000;
		struct feature_node *x = (struct feature_node *) malloc(5000*sizeof(struct feature_node));
		SparseMat_<float> sparseData ;
		this->album->database[not_selected[random_index[i]]].second.getSparseFeature().copyTo( sparseData )  ;
		int t = 0 ;
		for (cv::SparseMatIterator_<float> it = sparseData.begin(); it != sparseData.end(); it++ )
		{
			const SparseMat::Node *n = it.node() ;
			x[t].index = n->idx[0] + 1 ;
			x[t].value = it.value<float>() ;
			++t ;
		}

//#pragma omp parallel for reduction( max : max_dis ) num_threads( thread_num )
		for(size_t j = 0; j<satisfy_classifier_label.size(); ++j)
		{
			float temp_dis = currentClassifiers[j].getDis( x, t );
			if(temp_dis > max_dis)
			{
				max_dis = temp_dis;
			}
		}

		free( x ) ;
		index_dis.push_back(make_pair<float, int>(max_dis, not_selected[random_index[i]]));
	}

	for(map<int, PersonClassifier>::iterator model_it=this->classifiers_map.begin(); model_it != this->classifiers_map.end(); ++model_it)
	{
		if(model_it->second.isTrained())
			model_it->second.deleteSVM();
	}

	std::sort(index_dis.begin(), index_dis.end(), ascending );		//index_dis

	int uncertain_size = al_random_select_num ;
	vector<Index_Photo> uncertains;
	for(int i=0; i<uncertain_size; ++i){
		uncertains.push_back(this->album->database[index_dis[i].second]);
	}

	Mat allFeas = getFea(uncertains, this->config->get_fea_dim());



	int al_select = min( (int)not_selected.size(), this->config->get_al_select_per_iter() );


	int selected = 0;

	while(selected < al_select) {
		int queryLabel = this->queryLabel(this->album->database[index_dis[selected].second].second);
		this->album->database[index_dis[selected].second].second.isQueryLabel = true;
		this->album->database[index_dis[selected].second].second.setAssignLabel(queryLabel);

		map<int, PersonClassifier>::iterator it = this->classifiers_map.find(queryLabel);

		if ( it == this->classifiers_map.end() )
		{
			cout << "New Category is annotated!" << endl ;
			initNewClassifierWithOneSmp( this->album->database[index_dis[selected].second], queryLabel ) ;
		}
		else
		{
			it->second.addAlPositives(this->album->database[index_dis[selected].second]);
		}
		this->album->setSelected(index_dis[selected].second);
		selected ++;
	}


	// consider diversity
	cout << "AL selected number is: " << selected << endl;
	cout << "AL done ." <<endl<<endl;
}

void Solver::AL_low_spl_selectFrequency() {
	cout << "start special al..." << endl;
	int threshold = 0;
	cin >> threshold;
	vector<int> spl_selected_times;
	this->album->get_spl_selected_times(spl_selected_times);
	int count = 0;
	for (size_t i = 0; i < spl_selected_times.size(); ++i) {
		if ( this->album->database[spl_selected_times[i]].second.isQueryLabel == false && spl_selected_times[i] < threshold) {
			int query_label = this->queryLabel( this->album->database[i].second );
			this->album->database[i].second.isQueryLabel = true;
			this->album->database[i].second.setAssignLabel(query_label);
			this->album->setSelected( spl_selected_times[i] );
			count++;
		}
	}
	cout << "Special AL selected number is: " << count << endl;
	cout << "Special AL done." << endl;
}

void Solver::testTrainedClassifiers(){
	map<int, PersonClassifier> currClassifiers ;
	for(map<int, PersonClassifier>::iterator model_it=this->classifiers_map.begin(); model_it != this->classifiers_map.end(); ++model_it)
		if ( true )
		{
			model_it->second.loadXMl() ;
			currClassifiers.insert( *model_it );
		}

	int validation_set_size = 0;
	float accuracy = 0;
	//float pos_dis_num = 0;
	//float pos_acc = 0;
	for(map<int, vector<Photo> >::iterator vali_it=this->album->validation_set.begin(); vali_it != this->album->validation_set.end(); ++ vali_it)
	{
		int photo_size = vali_it->second.size();
		//float accuracy6 = 0;
		map<int, PersonClassifier>::const_iterator model_it = currClassifiers.find(vali_it->first);
		if(model_it != currClassifiers.end())
		{
#pragma omp parallel for reduction( + : accuracy ) num_threads( thread_num )
			for( int i=0; i< photo_size; ++i)
			{
				struct feature_node *x = (struct feature_node *) malloc(5000*sizeof(struct feature_node));
				SparseMat_<float> sparseData ;
				vali_it->second[i].getSparseFeature().copyTo( sparseData )  ;
				int j = 0 ;
				for (cv::SparseMatIterator_<float> it = sparseData.begin(); it != sparseData.end(); it++ )
				{
					const SparseMat::Node *n = it.node() ;
					x[j].index = n->idx[0] + 1 ;
					x[j].value = it.value<float>() ;
					++j ;
				}

				float max_dis = -1000;
				int max_label = -1;
				for(map<int, PersonClassifier>::iterator classifier_i =currClassifiers.begin(); classifier_i != currClassifiers.end(); ++classifier_i)
				{
					float tmp_dis = classifier_i->second.getDis( x, j );

					if(tmp_dis > max_dis)
					{
						max_dis = tmp_dis;
						max_label = classifier_i->first;
					}
				}
				if(max_label == vali_it->first) {
					accuracy++;
					//accuracy6++;
				}

				/*if (max_dis > 0) {
					pos_dis_num++;
					if (max_label == vali_it->first)
						pos_acc++;
				}*/

				free( x ) ;
			}

			validation_set_size += photo_size;
			//cout << "accuracy for " << vali_it->first << "th is " << accuracy6/photo_size << endl;

		}

	}

	for(map<int, PersonClassifier>::iterator model_it=currClassifiers.begin(); model_it != currClassifiers.end(); ++model_it)
	{
		model_it->second.deleteSVM();
	}
	//cout << this->iter << " " << max_acc_class << " " << max_accuracy << endl;

	if ( 0 == validation_set_size )
	{
		cout << "Please train the classifiers first before the testing phase!" << endl;
	}
	else
	{
		LOG << "accuracy: " << accuracy / validation_set_size << " for " << validation_set_size << " samples" << end;
		LOG << "queried photos: " << this->album->getQueriedNum() <<end;
		LOG << "queried photos percent: " << (double)this->album->getQueriedNum() / this->album->database.size() <<end;
		//LOG << "positives accuracy: " << pos_acc / validation_set_size << end;
		//LOG << "positives percent: " << pos_dis_num / validation_set_size << end;
		LOG << "selected photos: "<<this->album->getSelectedNum()<<end;
		LOG << "selected photos percent: " << (double)this->album->getSelectedNum() / this->album->database.size() <<end;
		LOG << "current unclear sample: " << this->album->getUnclearNum() << end ;

		ofstream accuracy_out ;
		accuracy_out.open( "cacd_accuracy.txt", ios::out | ios::app ) ;
		accuracy_out << (double)this->album->getQueriedNum() / this->album->database.size() << " " << accuracy / validation_set_size << endl ;
		accuracy_out.close() ;
	}
}


void getSeeds(vector<Index_Photo>& cluster, float* center_fea, int fea_dim, vector<Index_Photo>& seeds){
	int index_1 = 0;
	int index_2 = 0;

	float distance_1 = 100;
	float distance_2 = 100;

	int size = cluster.size();

	float dis = 0;
	for(int i=0; i<size; ++i){
		dis = distance(cluster[i].second.getFeature().ptr<float>(0), center_fea, fea_dim);
		if(dis < distance_1){
			index_1 = i;
			distance_1 = dis;
		}
		else if(dis < distance_2){
			index_2 = i;
			distance_2 = dis;
		}
	}
	seeds.push_back(cluster[index_1]);
	seeds.push_back(cluster[index_2]);
}


void Solver::verify()
{
	cout << endl << "Begin Verification..." << endl ;
	bool need2verify = false ;
	vector<PersonClassifier> currClassifiers ;
	vector<int> currClassifierLabels ;
	for(map<int, PersonClassifier>::iterator it=this->classifiers_map.begin(); it != this->classifiers_map.end(); ++it)
		if ( it->second.isTrained() )
		{
			it->second.loadXMl() ;
			currClassifiers.push_back(it->second) ;
			currClassifierLabels.push_back(it->first) ;
		}

	float accuracy = 0;
	float sum = 0;
	struct feature_node *x = (struct feature_node *) malloc(5000*sizeof(struct feature_node));
	for( size_t classifier_i = 0; classifier_i < currClassifiers.size(); classifier_i++ )
	{
		vector<Index_Photo>& posSmps = currClassifiers[classifier_i].getAlPositives() ;
		//vector<Index_Photo>& t = currClassifiers[classifier_i].get
		for ( size_t i = 0; i < posSmps.size(); i++ )
		{
			SparseMat_<float> sparseData ;
			posSmps[i].second.getSparseFeature().copyTo( sparseData ) ;
			int offset = 0 ;
			for (cv::SparseMatIterator_<float> it = sparseData.begin(); it != sparseData.end(); it++ )
			{
				const SparseMat::Node *n = it.node() ;
				x[offset].index = n->idx[0] + 1 ;
				x[offset].value = it.value<float>() ;
				++offset ;
			}

			vector<float> dis_list( currClassifiers.size() ) ;
#pragma omp parallel for num_threads( thread_num )
			for(size_t j = 0; j < currClassifiers.size(); ++j)
			{
				dis_list[j] = currClassifiers[j].getDis( x, offset );
			}

			int sum_dis = 0 ;
			float max_dis = -1 ;
			int predicted_label = -1 ;
			for(size_t j = 0; j < currClassifiers.size(); ++j)
			{
				sum_dis += ( dis_list[j] > 0 ) ? 1 : 0 ;

				if ( max_dis < dis_list[j] )
				{
					max_dis = dis_list[j] ;
					predicted_label = currClassifierLabels[j] ;
				}
			}

			// posSmps[i].first: index
			// currClassifierLabels[classifier_i): label
			if ( ( predicted_label != currClassifierLabels[classifier_i] ) || sum_dis != 1 ) {
				need2verify = true ;
				map<int, PersonClassifier>::iterator it = this->classifiers_map.find(currClassifierLabels[classifier_i]);
				it->second.add_al_incorrect_time(posSmps[i].first);

				if (it->second.get_al_incorrect_times(posSmps[i].first) > 3) {
					this->album->setNotSelected(posSmps[i].first);
					it->second.pop_al_positives(posSmps[i].first);
				}
			}

			if ( predicted_label != currClassifierLabels[classifier_i] )
			{
				cout << " PhotoName: " << posSmps[i].second.getName() << "--Incorrectly Classified! Estimated as " << predicted_label << "-th category while ground truth is " << currClassifierLabels[classifier_i] << "-th category!" << endl ;

			} else if ( sum_dis == 0 )
			{
				cout << " PhotoName: " << posSmps[i].second.getName() << "--Not Recognized!" << endl ;				
			} else {
				accuracy++;
			}

			if ( sum_dis > 1 )
			{
				cout << " PhotoName: " << posSmps[i].second.getName() << " --Ambiguity Sample for the  " ;
				for ( size_t j = 0; j < currClassifiers.size(); ++j )
				{
					if ( dis_list[j] > 0 )
						cout << j << "-th " << "with dis: " << dis_list[j] << ", ";
				}
				cout << " classifiers!" << endl ;				
			}
		}
		sum += posSmps.size();
	}

	if ( need2verify )
	{

		cout << "Please update the training annotations in " << endl ;
		cout << this->config->get_init_id_path() << endl ;
		cout << this->config->get_train_id_path() << endl ;

		cout << "Then press any key to reload all annotations to continue!" << endl << endl ;
//		cin.get() ;

		this->album->reLoadLabels( this->config ) ;
	}
	cout << "accuracy for train set is: " << accuracy / sum << endl;
	cout << "Verification Done!" << endl ;
}

void Solver::initNewClassifierWithOneSmp( Index_Photo& smp, const int CategoryLabel )
{
	PersonClassifier* classifier = new PersonClassifier(this->config);
	classifier->setPersonLabel(CategoryLabel);
	classifier->addAlPositives(smp);
	this->classifiers_map.insert(pair<int, PersonClassifier>(CategoryLabel, *classifier));

	//calculate distance to all unclear samples
	float avg_dist = 0 ;
	int totalCount = 0 ;
	vector<float> dists ;
	for ( list<Index_Photo>::iterator it = this->unclear_set.begin(); it != this->unclear_set.end(); it++ )
	{
		float distance = smp.second.getDistance( it->second ) ;
		dists.push_back( distance ) ;
		avg_dist += distance ;
		totalCount ++ ;
	}
	avg_dist /= totalCount ;

	int count = 0 ;
	list<Index_Photo>::iterator it_ori = this->unclear_set.begin() ;
	for ( size_t i = 0; i < dists.size(); i++  )
	{
		list<Index_Photo>::iterator it = it_ori ;
		it_ori++ ;
		if ( avg_dist > dists[i] * 3 )
		{
			int queryLabel = this->queryLabel( it->second ) ;
			this->album->database[it->first].second.isQueryLabel = true ;
			this->album->setSelected( it->first ) ;
			this->album->setClear( it->first ) ;

			map<int, PersonClassifier>::iterator classifier_it = this->classifiers_map.find(queryLabel);

			if ( classifier_it == this->classifiers_map.end() )
			{
				PersonClassifier* classifier = new PersonClassifier(this->config);
				classifier->setPersonLabel(queryLabel);
				classifier->addAlPositives( *it ) ;
				this->classifiers_map.insert(pair<int, PersonClassifier>(queryLabel, *classifier));
			}
			else
				classifier_it->second.addAlPositives( *it ) ;

			if ( CategoryLabel == queryLabel )
			{
				count++ ;
			}

			this->unclear_set.erase( it ) ;
		}

		if ( count >= 5 )
		{
			break ;
		}
	}
	cout << "Initialize " << count << " samples for " << CategoryLabel << "-th classifier from the Unclear Set with " << totalCount << " samples" << endl ;
}

bool train_cmp(pair<Index_Photo, float> a, pair<Index_Photo, float> b) { return a.second > b.second;};

void Solver::trainAllClassifiersOneVsALL(){
	vector<PersonClassifier> tmpC;
	for(map<int, PersonClassifier>::iterator it=this->classifiers_map.begin(); it != this->classifiers_map.end(); ++it)
		if ( it->second.isInitialized() )
		{
			if ( it->second.isTrained() )
				it->second.loadXMl() ;

			tmpC.push_back(it->second) ;
		}

	int size_all_classifier = tmpC.size() ;
	if ( size_all_classifier == 0 )
	{
		cout << "No classifiers are initialized! Please check!" << endl ; ;
		return ;
	}


# pragma omp parallel for default(none) shared(tmpC, size_all_classifier) num_threads( thread_num )
	for(int i=0; i<size_all_classifier; ++i){
		map<int, PersonClassifier>::iterator it = this->classifiers_map.find(tmpC[i].getPersonLabel());
		vector<Index_Photo> negatives;
		int totalSmpNum = 1 ;
		float AmbiguitySmpNum = 0 ;
		vector<pair<Index_Photo, float> > conf_vector;
		for(int j=0; j<size_all_classifier; ++j){
			if(it->second.getPersonLabel() != tmpC[j].getPersonLabel())
			{
				vector<Index_Photo> tmpP;
				tmpC[j].getAllPositives(tmpP);

				for(int z = 0; z < (int)tmpP.size(); ++z)
				{
					/*if ( it->second.isTrained() )
					{
						const float confVal = it->second.getDis( tmpP[z].second ) ;

						if ( confVal > -1.001 ){
							negatives.push_back(tmpP[z]);
							AmbiguitySmpNum += 1 ;
						}


						totalSmpNum++ ;
					}
					else
					{
						negatives.push_back(tmpP[z]);
					}*/
					negatives.push_back(tmpP[z]);
				}
				/*if (it->second.isTrained())
					negatives.push_back(tmpP[max_index]);*/
			}
		}

		it->second.trainModel(negatives, this->config);
		it->second.setAvgAccuracy( 1.f - AmbiguitySmpNum / totalSmpNum ) ;
	}
	//cout << "Trained Classifier Number: " << tmpC.size() << endl ;
}


void Solver::trainAllClassifiersTogether( float C ){
//	cout << "start training all satisfied classifier ..."<<endl;

	vector<PersonClassifier> tmpC;
	for(map<int, PersonClassifier>::iterator it=this->classifiers_map.begin(); it != this->classifiers_map.end(); ++it){
		if ( it->second.isInitialized() )
			tmpC.push_back(it->second);
	}

	int size_all_classifier = tmpC.size();
//	cout << "Total Classifier Num: " << size_all_classifier << endl ;

	vector<SparseMat> features ;
	vector<int> labels ;
	vector<float> weights ;

	for(int i=0; i<size_all_classifier; ++i)
	{
		vector<Index_Photo> tmp_pos ;
		vector<float> pos_weights ;
		tmpC[i].getAllPositives( tmp_pos, pos_weights ) ;

		for ( size_t j = 0; j < tmp_pos.size(); j++ )
		{
			features.push_back( tmp_pos[j].second.getSparseFeature() ) ;
			labels.push_back( tmpC[i].getPersonLabel() ) ;
			weights.push_back( pos_weights[j] ) ;
		}
	}

	float* train_labels = new float[labels.size()] ;
	float* train_weights = new float[labels.size()] ;
	for ( size_t i = 0; i < labels.size(); i++ )
	{
		train_labels[i] = labels[i] ;
		train_weights[i] = 1; //  weights[i] ;
	}

	char modelSavedPath[100] ;
	sprintf( modelSavedPath, ".//model//all_model.xml" );

	std::ostringstream ss;
	ss << C ;
	std::string strC(ss.str());

	int argc = 10; //0 -- 0.345904 1--0.30 2--0.311737 3--0.31 4--0.32 5--
	const char* argv[] = {"", "-s", "2", "-C", "0.001", "-B", "10", "-q", "None", modelSavedPath}; //"-W", "inside", "-q", "None", modelSavedPath} ;
	main_train( argc, argv, features, train_weights, train_labels ) ;

	delete [] train_labels ;
	delete [] train_weights ;

//	cout << "train all satisfied classifier done."<<endl<<endl;;
}

void Solver::testTrainedClassifiersTogether()
{
	PersonClassifier allClassifier( ".//model//all_model.xml" ) ;

	float accuracy = 0 ;
	int validation_set_size = 0 ;
	double* dec_val = new double[classifiers_map.size()] ;
	for(map<int, vector<Photo> >::iterator vali_it=this->album->validation_set.begin(); vali_it != this->album->validation_set.end(); ++ vali_it)
	{
		size_t photo_size = vali_it->second.size() ;

//#pragma omp parallel for reduction( + : accuracy ) shared( vali_it ) num_threads( thread_num )
		for(size_t i=0; i< photo_size; ++i)
		{
			int predict_label = allClassifier.predictLabel( vali_it->second[i], dec_val ) ;
			LOG << "predict label " << predict_label << end;
			accuracy += ( predict_label == vali_it->first ? 1 : 0 ) ;
		}
		validation_set_size += photo_size;
		break;
	}
	delete [] dec_val ;

	LOG << "Together accuracy: " << accuracy / validation_set_size << end;
	LOG << "Together queried photos: " << this->album->getQueriedNum() <<end;
	LOG << "Together queried photos percent: " << (double)this->album->getQueriedNum() / this->album->database.size() <<end;
	LOG << "Together selected photos: "<<this->album->getSelectedNum()<<end;
	LOG << "Together selected photos percent: " << (double)this->album->getSelectedNum() / this->album->database.size()  <<end;
}

void Solver::print_spl_accuracy() {
	float sum_accuracy = 0;
	float sum_size = 0;
	for(map<int, PersonClassifier>::iterator it=this->classifiers_map.begin(); it!=this->classifiers_map.end(); ++it) {
		vector<int> spl_index ;
		it->second.getSplPositivesDatabaseIndex(spl_index);
		float accuracy = 0;
		for ( size_t i = 0; i < spl_index.size(); i++ )
		{
			if (this->album->database[spl_index[i]].second.getAssignedLabel() == this->album->database[spl_index[i]].second.getTrueLabel()) {
				accuracy++;
			}
		}
		sum_accuracy += accuracy;
		sum_size += spl_index.size();
		cout << "SPL num for " << it->first << "th is " << spl_index.size() << " with accuracy " << accuracy / spl_index.size() << endl;
	}
	cout << "SPL select number is " << sum_size << endl;
	cout << "SPL accuracy is " << sum_accuracy / sum_size << endl;
}

