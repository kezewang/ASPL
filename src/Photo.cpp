/*
 * Photo.cpp
 *
 */

#include "Photo.hpp"

#include <cmath>

using namespace std;

typedef pair<int, float> PAIR; // first: index, second: distance

bool cmpPAIR(const PAIR& left, const PAIR& right){
	return (left.second < right.second);
}

float Photo::getDistance(Photo& p){
	const Mat& p_featMat = p.getFeature();
	const Mat& featMat = this->getFeature() ;

	return  norm( p_featMat, featMat, NORM_L2 ) ;
}


/*
vector<int> Photo::retrieveSimilarPhotos(const vector<Photo>& database){
	cout << "retrive similar photos ... " << endl;

	Photo cur_photo;
	vector<PAIR> distance_pair;
	float tmp_dis = 0.0;
	int database_size = database.size();
	for (int db_index = 0; db_index < database_size; db_index++){
		tmp_dis = getDistance(database[db_index]);
		distance_pair.push_back(make_pair(db_index, tmp_dis));
	}

	sort(distance_pair.begin(), distance_pair.end(), cmpPAIR);

	vector<int> retrive_index; // index of retrieve result
	int pair_size = distance_pair.size();
	for (int i = 0; i < pair_size; i++){
		retrive_index.push_back(distance_pair[i].first);
	}

	return retrive_index;
}
*/

bool Photo::hasAssingedLabel(){
	return (assigned_label != -1);
}

bool Photo::isAssignLabelCorrect(){
	return (assigned_label != -1 && true_label == assigned_label); // label is assigned && is correct
}
/*
const vector<int> Photo::getNegLabel(){
	return neg_label;
}

void Photo::addNegLabel(int neg){
	neg_label.push_back(neg);
}
*/
void Photo::setFeature(SparseMat tmp_feature){
	this->feature = tmp_feature;
}

void Photo::setFeature(Mat tmp_feature){
	this->feature = tmp_feature ;
}

const SparseMat Photo::getSparseFeature(){
	return feature;
}

const Mat Photo::getFeature() {
	Mat denseFeat ;
	feature.copyTo( denseFeat ) ;
	return denseFeat;
}

const string Photo::getName(){
	return name;
}

void Photo::setName(string s){
	name = s;
}

const int Photo::getTrueLabel(){
	return true_label;
}

void Photo::setTrueLabel(int n){
	true_label = n;
}

int Photo::getAssignedLabel(){
	return assigned_label;
}

void Photo::setAssignLabel(int tmp_label){
	assigned_label = tmp_label;
}


Photo::Photo(Config* config){
	name = "";
	true_label = LABEL_NOT_ASSIGNED;
	assigned_label = LABEL_NOT_ASSIGNED; // -1 means haven't assigned label
	//neg_label.clear();
	fea_dim = config->get_fea_dim();
	int dim_size[] = { fea_dim } ;
	feature = SparseMat( 1, dim_size, CV_32F ) ; //Mat::zeros(1, fea_dim, CV_32F);
	this->isQueryLabel = false;
}


