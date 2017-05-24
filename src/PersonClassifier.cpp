/*
 * PersonClassifier.cpp
 *
 */
#include <stdlib.h>
#include "train.h"
#include "PersonClassifier.h"

PersonClassifier::PersonClassifier(Config* config){
	this->label = LABEL_NOT_ASSIGNED;
	//this->classifier = new CvSVM();
	this->trained = false;
	this->initialization = false ;
	//	params.svm_type    = CvSVM::C_SVC;
	//	params.C = 10;
	//	params.nu = 0.5;
	//	params.kernel_type = CvSVM::LINEAR;
	this->model_ = NULL ;
	this->spl_dis_threshold_decrease_rate = 0.08 ;
	this->spl_dis_threshold = 0.12 ;
	this->avgAccuracy = 0 ;
	//	classifier = NULL ;
	current_po_size = 0 ;
	current_al_po_size = 0 ;

	//	params.term_crit   = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	this->above_average = false;
}

PersonClassifier::PersonClassifier(const char* modelSavedPath)
{
	this->label = LABEL_NOT_ASSIGNED;
	this->trained = true;
	this->initialization = true ;
	current_po_size = 0 ;
	current_al_po_size = 0 ;
	this->spl_dis_threshold_decrease_rate = 0.8 ;
	this->spl_dis_threshold = 0.12 ;
	this->avgAccuracy = 0 ;

	this->above_average = false;
	this->model_ = load_model( modelSavedPath ) ;
}

void PersonClassifier::getAllPositives(vector<Index_Photo>& result){
	int size = this->al_positives.size();
	//cout << "al_positives num: " << size << endl;
	for(int i=0; i<size; ++i)
		result.push_back(this->al_positives[i]);

	size = this->spl_positives.size();
	//cout << "spl_positives num: " << size << endl;
	for(int i=0; i<size; ++i)
		result.push_back(this->spl_positives[i]);
}

void PersonClassifier::getAllPositives(vector<Index_Photo>& result, vector<float>& weights)
{
		for(size_t i=0; i < this->al_positives.size(); ++i)
		{
			result.push_back(this->al_positives[i]);
			weights.push_back( 1.f ) ;
		}

		for(size_t i=0; i < this->spl_positives.size(); ++i)
		{
			result.push_back(this->spl_positives[i]);
			weights.push_back( this->spl_weight[i] ) ;
		}
}

void PersonClassifier::getAlPositivesDatabaseIndex(vector<int>& result){
	result.clear() ;
	for(int i=0; i< (int)this->al_positives.size(); ++i)
		result.push_back(this->al_positives[i].first);
}

void PersonClassifier::getSplPositivesDatabaseIndex(vector<int>& result){
	int size = this->spl_positives.size();
	for(int i=0; i<size; ++i)
		result.push_back(this->spl_positives[i].first);
}

void PersonClassifier::getAllPositivesNames(vector<string>& names){
	int size = this->al_positives.size();
	for(int i=0; i<size; ++i)
		names.push_back(this->al_positives[i].second.getName());
	size = this->spl_positives.size();
	for(int i=0; i<size; ++i)
		names.push_back(this->spl_positives[i].second.getName());
}

void PersonClassifier::getALPositivesNames(vector<string>& names){
	int size = this->al_positives.size();
	for(int i=0; i<size; ++i)
		names.push_back(this->al_positives[i].second.getName());
}

void PersonClassifier::getSPLPositivesNames(vector<string>& names){
	int size = this->spl_positives.size();
	for(int i=0; i<size; ++i)
		names.push_back(this->spl_positives[i].second.getName());
}

void PersonClassifier::AlSplVerification(Photo& verifier, vector<int>& put_back_index, Config* config){
	int spl_size = this->spl_positives.size();
	const float al_spl_verification_dis_threshold = config->get_al_spl_verification_dis_threshold();
	vector<Index_Photo> new_spl_positives;
	for(int i=0; i<spl_size; ++i){
		if(this->spl_positives[i].second.getDistance(verifier) >= al_spl_verification_dis_threshold)
			new_spl_positives.push_back(this->spl_positives[i]);
		else
			put_back_index.push_back(this->spl_positives[i].first);
	}
	this->spl_positives = new_spl_positives;
}

bool PersonClassifier::isPositive( Photo& photo)
{
	assert(this->trained == true);
	return this->predictLabel( photo ) == POSITIVE_LABEL;
}

Mat getFeatureMat(vector<Index_Photo>& positives, vector<Index_Photo>& negatives, int fea_dim, float* train_labels){
	assert(train_labels != NULL);

	int po_size = positives.size();
	int ne_size = negatives.size();
	int label_size = po_size + ne_size;

	Mat features(label_size, fea_dim, CV_32F);

	float *fea_ptr;
	for(int i=0; i<po_size; ++i){
		fea_ptr = features.ptr<float>(i);
		const float *mat_ptr = positives[i].second.getFeature().ptr<float>(0);
		for(int j=0; j<fea_dim; ++j)
			fea_ptr[j] = mat_ptr[j];
		train_labels[i] = POSITIVE_LABEL;
	}
	for(int i=0; i<ne_size; ++i){
		fea_ptr = features.ptr<float>(i + po_size);
		const float *mat_ptr = negatives[i].second.getFeature().ptr<float>(0);
		for(int j=0; j<fea_dim; ++j)
			fea_ptr[j] = mat_ptr[j];
		train_labels[i + po_size] = NEGATIVE_LABEL;
	}
	return features;
}

void getFeatureSparseMat(vector<SparseMat>& features, vector<Index_Photo>& positives, vector<Index_Photo>& negatives, int fea_dim, float* train_labels){
	assert(train_labels != NULL);

	int po_size = (int)positives.size();
	int ne_size = (int)negatives.size();
	int label_size = po_size + ne_size;

	features = vector<SparseMat>( label_size ) ;

	for(int i=0; i<po_size; ++i){
		features[i] = positives[i].second.getSparseFeature() ;
		train_labels[i] = POSITIVE_LABEL;
	}
	for(int i=0; i<ne_size; ++i){
		features[i + po_size] = negatives[i].second.getSparseFeature() ;
		train_labels[i + po_size] = NEGATIVE_LABEL;
	}
}

//bool PersonClassifier::isPositive(Photo& photo){
//	assert(this->istrained == true);
//	return this->classifier->predict(photo.getFeature()) == POSITIVE_LABEL ;
//}

void PersonClassifier::deleteSVM(){
	//		delete this->classifier;
	free_and_destroy_model(&this->model_);
}

float PersonClassifier::predictLabel( Photo& photo )
{
	int nr_feature=get_nr_feature(model_);
	int nr_class=get_nr_class(this->model_);
	struct feature_node *x = (struct feature_node *) malloc(5000*sizeof(struct feature_node));
	SparseMat_<float> sparseData ;
	photo.getSparseFeature().copyTo( sparseData )  ;
	int j = 0 ;
	for (cv::SparseMatIterator_<float> it = sparseData.begin(); it != sparseData.end(); it++ )
	{
		const SparseMat::Node *n = it.node() ;
		x[j].index = n->idx[0] + 1 ;
		x[j].value = it.value<float>() ;
		++j ;
	}
	if(model_->bias>=0)
	{
		x[j].index = nr_feature+1 ;
		x[j].value = model_->bias ;
		j++ ;
	}
	x[j++].index = -1;

	double* dec_val = new double [nr_class] ;
	int predict_label = (int)predict_values(this->model_, x, dec_val);

	delete [] dec_val ;
	free( x ) ;
	return predict_label ;
}

float PersonClassifier::predictLabel( Photo& photo, double* dec_val)
{
	int nr_feature=get_nr_feature(model_);
	struct feature_node *x = (struct feature_node *) malloc(5000*sizeof(struct feature_node));

	SparseMat_<float> sparseData ;
	photo.getSparseFeature().copyTo( sparseData )  ;
	int j = 0 ;
	for (cv::SparseMatIterator_<float> it = sparseData.begin(); it != sparseData.end(); it++ )
	{
		const SparseMat::Node *n = it.node() ;
		x[j].index = n->idx[0] + 1 ;
		x[j].value = it.value<float>() ;
		++j ;
	}

	if(model_->bias>=0)
	{
		x[j].index = nr_feature+1 ;
		x[j].value = model_->bias ;
		j++ ;
	}
	x[j++].index = -1;

	int predict_label = (int)predict_values(this->model_, x, dec_val);
	free( x ) ;
	return predict_label ;
}

float PersonClassifier::getDis( struct feature_node* x, const int j )
{
	int nr_feature=get_nr_feature(model_);
	int nr_class=get_nr_class(this->model_);
	double* decision_value = (double *) malloc(nr_class*sizeof(double));

	if(model_->bias >= 0)
	{
		x[j].index = nr_feature+1 ;
		x[j].value = model_->bias ;
	}
	x[j].index = -1;

	predict_values(this->model_, x, decision_value);
	//cout << decision_value[0] << " " << decision_value[1] << endl;
	//float dec_val = decision_value[0] > decision_value[1] ? decision_value[0] : decision_value[1];
	float confVal = (float)decision_value[0] ;
	//float confVal = dec_val;
	free( decision_value ) ;
	return confVal ;
}

float PersonClassifier::getDis(Photo& photo){
	//		return this->classifier->predict(photo.getFeature(), true);
	//cout << model_ << endl;
	int nr_feature=get_nr_feature(model_);
	struct feature_node *x = (struct feature_node *) malloc(5000*sizeof(struct feature_node));
	int nr_class=get_nr_class(this->model_);
	double* decision_value = (double *) malloc(nr_class*sizeof(double));

	SparseMat_<float> sparseData ;
	photo.getSparseFeature().copyTo( sparseData )  ;
	int j = 0 ;
	for (cv::SparseMatIterator_<float> it = sparseData.begin(); it != sparseData.end(); it++ )
	{
		const SparseMat::Node *n = it.node() ;
		x[j].index = n->idx[0] + 1 ;
		x[j].value = it.value<float>() ;
		++j ;
	}

	if(model_->bias>=0)
	{
		x[j].index = nr_feature+1 ;
		x[j].value = model_->bias ;
		//j++ ;
	}
	x[j++].index = -1;

	predict_values(this->model_, x, decision_value);
	float confVal = (float)decision_value[0] ;
	free(decision_value);
	free( x ) ;
	return confVal ;
}

void PersonClassifier::loadXMl(){
	//		this->classifier = new CvSVM();
	//		ostringstream xmlname;
	//		xmlname <<  ".//model//" << this->label << "_model.xml";
	//		this->classifier->load(xmlname.str().c_str());

	char modelSavedPath[100] ;
	sprintf( modelSavedPath, ".//model//%d_model.xml", this->label );
	this->model_ = load_model( modelSavedPath ) ;
}

void PersonClassifier::loadXMl(char *path) {
	this->model_ = load_model(path);
}

void PersonClassifier::trainModel(vector<Index_Photo> negatives, Config* config){
	const int ne_size = negatives.size();
	int po_size = this->getAllPositiveNum();
	int label_size = po_size + ne_size;
	float *train_labels = new float[label_size];

	vector<float> allWeight ;
	vector<Index_Photo> allPos;
	this->getAllPositives(allPos, allWeight);



	//	Mat features = getFeatureMat(allPos, negatives, config->get_fea_dim(), train_labels);
	vector<SparseMat> features ; 
	getFeatureSparseMat( features, allPos, negatives, config->get_fea_dim(), train_labels ) ;

	float* weight = new float[label_size] ;

	for( size_t i = 0; i < allWeight.size(); i++ )
		weight[i] = allWeight[i] ;
	for ( int i = (int) allWeight.size(); i < label_size; i++ )
		weight[i] = 1 ;

	//	cout <<"pos size: "<<po_size<<", neg size:"<<ne_size<<", mat row: "<<features.rows<<endl;
	//	cout << "start training model for label: " << this->label <<" ..." << endl;
	//	int e1 = cv::getTickCount();
	/////  svm  train    /////

	char modelSavedPath[100] ;
	sprintf( modelSavedPath, ".//model//%d_model.xml", this->label );
	char weightFile[100];
	sprintf( weightFile, ".//weight//%d_weight.xml", this->label );

	int argc = 16;
	const char* argv[] = {"", "-s", "2", "-c", "0.001", "-w0", "5", "-w1", "1",  "-W", "inside", "-q", "None", modelSavedPath} ;

	main_train( argc, argv, features, weight, train_labels ) ;

	delete [] weight ;
	delete [] train_labels ;
	this->trained = true;
}

const float PersonClassifier::get_spl_dis_threshold(){ return this->spl_dis_threshold; }
void PersonClassifier::calculate_spl_dis_threshold() {
    this->loadXMl() ;
   // cout << "calculate_spl_dis_threshold" << endl;
	const int size = this->al_positives.size();
	if (size <= 0) {
		this->spl_dis_threshold = 0.9;
                cout << "size equals 0" << endl;
                return ;
		//return 0.9;
	}
	struct feature_node *x = (struct feature_node *) malloc(5000*sizeof(struct feature_node));
	float min_dis = 10000;
	//cout << "dis for" << this->label << "th al: ";
	for (int i = 0; i < size; i++) {
		SparseMat_<float> sparseData;
		this->al_positives[i].second.getSparseFeature().copyTo( sparseData );
		int offset = 0;
		for (cv::SparseMatIterator_<float> it = sparseData.begin(); it != sparseData.end(); it++ )
		{
			const SparseMat::Node *n = it.node() ;
			x[offset].index = n->idx[0] + 1 ;
			x[offset].value = it.value<float>() ;
			++offset ;
		}
		//dis_sum += (1 - this->getDis(x, offset));
		float dis = this->getDis(x, offset);
		if (dis < min_dis) {
			min_dis = dis;
		}
		//cout << dis << "\t";
	}
	//cout << endl;
	free(x);
	//this->spl_dis_threshold = dis_sum / size;
	this->spl_dis_threshold = min_dis;
    this->deleteSVM() ;
}

void PersonClassifier::remove_spl_low_dis_index(vector<int>& result){
	int size = this->spl_positives.size();
	vector<Index_Photo> new_spl_positive;
	vector<float> new_spl_weight;
	vector<float> old_spl_dis;
	float sum_dis = 0;
	for (int i = 0; i < size; ++i) {
		float dis = this->getDis(this->spl_positives[i].second);
		old_spl_dis.push_back(dis);
		sum_dis += dis;
	}
	//float avg_dis = sum_dis / size;
	//cout << this->label << "th remove stage: " << endl;
	float avg_dis = this->get_spl_dis_threshold();
	//cout << avg_dis << endl;
	for (int i = 0; i < size; ++i) {
		//cout << this->spl_positives[i].second.getTrueLabel() <<"-"<< old_spl_dis[i] << " ";
		if (old_spl_dis[i] > avg_dis) {
			new_spl_weight.push_back(this->spl_weight[i]);
			new_spl_positive.push_back(this->spl_positives[i]);
		} else {
			result.push_back(this->spl_positives[i].first);
		}
	}
	//cout << endl;
	this->spl_positives = new_spl_positive;
	this->spl_weight = new_spl_weight;
}

//void PersonClassifier::trainModel(vector<Index_Photo> negatives, Config* config){
//	this->classifier = new CvSVM();
//
//	int po_size = this->getAllPositiveNum();
//	int ne_size = negatives.size();
//	int label_size = po_size + ne_size;
//	float *train_labels = new float[label_size];
//	vector<Index_Photo> allPos;
//	this->getAllPositives(allPos);
//
//	Mat features = getFeatureMat(allPos, negatives, config->get_fea_dim(), train_labels);
//	Mat labelsMat(label_size, 1, CV_32FC1, train_labels);
//
////	for ( int row = 0; row < features.rows; row++ )
////	{
////		const float* featPtr = features.ptr<float>( row ) ;
////		for ( int col = 0; col < features.cols; col++ )
////		{
////			cout << featPtr[col] << " " ;
////		}
////		cout << endl ;
////	}
//
////	cout <<"pos size: "<<po_size<<", neg size:"<<ne_size<<", mat row: "<<features.rows<<endl;
////	cout << "start training model for label: " << this->label <<" ..." << endl;
////	int e1 = cv::getTickCount();
//	/////  svm  train    /////
//	this->classifier->train(features, labelsMat, Mat(), Mat(), this->params);
////	int e2 = cv::getTickCount();
////	double elapse_time = (double)(e2 - e1)/ cv::getTickFrequency();
////	cout <<" svm training complete, time usage: "<<elapse_time<<endl;
//	ostringstream xmlname;
//	xmlname << ".//model//" << this->label << "_model.xml";
//
//	this->classifier->save(xmlname.str().c_str());
//
//	delete classifier;
//
////	cout << "training done ..." << endl << endl;
//	this->istrained = true;
//}
