/*
 * PersonClassifier.h
 *
 */

#include "Photo.hpp"
#include "Album.h"
#include <cv.h>
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "linear.h"

#ifndef PERSONCLASSIFIER_H_
#define PERSONCLASSIFIER_H_

typedef pair<int, int> AL_Wrong_Index;

const int POSITIVE_LABEL = 0;
const int NEGATIVE_LABEL = 1;

class PersonClassifier{
public:
//	PersonClassifier() {}
	PersonClassifier(Config* config);

	PersonClassifier( const char* modelSavedPath ) ;

	virtual ~PersonClassifier(){}

	int getPersonLabel(){ return this->label; }

	void setPersonLabel(int label){ this->label = label;}

	void trainModel(vector<Index_Photo> negatives, Config* config);

	void loadXMl() ;
	void loadXMl(char *path);

	void deleteSVM() ;

	float getDis(Photo& photo) ;

	float getDis( struct feature_node* x, const int j ) ;

	float predictLabel( Photo& photo ) ;

	float predictLabel( Photo& photo, double* dec_val) ;

	void addSplPositives(Index_Photo& photo, float weight = 1){ this->spl_positives.push_back(photo); this->spl_weight.push_back( weight ) ; }
	void popSplPositivesVector() {this->spl_positives.pop_back(); this->spl_weight.pop_back();}

	void addAlPositives(Index_Photo& photo)
	{
		this->al_positives.push_back(photo);
		this->al_incorrect_times.push_back(AL_Wrong_Index(photo.first, 0));

		if ( this->al_positives.size() >= 4 )
		{
			this->setInitialized( true ) ;
		}
	}

	bool isInitialized(){ return this->initialization; }
	void setInitialized( bool flag ) { this->initialization = flag; }
	bool isTrained() { return this->trained ; }
	void setTrained( bool flag ) { this->trained = flag ; }

	bool isPositive(Photo& photo);


	int getAllPositiveNum(){ return this->al_positives.size()+this->spl_positives.size(); }

	int getAlPositiveNum(){ return this->al_positives.size(); }

	int getSplPositiveNum(){ return this->spl_positives.size(); }

	void getAllPositives(vector<Index_Photo>& result);

	void getAllPositives(vector<Index_Photo>& result, vector<float>& weights) ;

	vector<Index_Photo>& getAlPositives(){ return this->al_positives; }

	void getAlPositivesDatabaseIndex(vector<int>& result);

	void getSplPositivesDatabaseIndex(vector<int>& result);

	void getAllPositivesNames(vector<string>& names);

	void getALPositivesNames(vector<string>& names);

	void getSPLPositivesNames(vector<string>& names);

	void get_AL_incorrect (vector<AL_Wrong_Index>& result) {
		result.clear() ;
		for(int i=0; i< (int)this->al_incorrect_times.size(); ++i)
			result.push_back(this->al_incorrect_times[i]);	
	}

	void setAlPositives(const vector<Index_Photo>& positives, const vector<AL_Wrong_Index>& wrong_times)
	{
		this->al_positives.clear() ;
		this->al_incorrect_times.clear();
		for ( int i = 0; i < (int) positives.size(); i++ )
		{
			this->al_positives.push_back( positives[i] ) ;
			//this->al_incorrect_times.push_back(AL_Wrong_Index(positives[i].first, 0));
			this->al_incorrect_times.push_back( wrong_times[i] );
		}
	}

	void setAvgAccuracy( float avgAccuracy ) { this->avgAccuracy = avgAccuracy ; }
	float getAvgAccuracy() const { return this->avgAccuracy ; }

	void setSplPositives(vector<Index_Photo>& positives){ this->spl_positives = positives; }

	void clearSplPositives(){ this->spl_positives = vector<Index_Photo>(); this->spl_weight = vector<float>(); }

	void clearAlPositives(){ this->al_positives = vector<Index_Photo>(); this->al_incorrect_times = vector<AL_Wrong_Index>();}

	void AlSplVerification(Photo& verifier, vector<int>& put_back_index, Config* config);

	void update_spl_dis_threshold(){
		this->spl_dis_threshold += this->spl_dis_threshold_decrease_rate * this->avgAccuracy ;
		this->spl_dis_threshold = ( this->spl_dis_threshold > 0.9 ) ? 0.9 : this->spl_dis_threshold ;
		//this->calculate_spl_dis_threshold();
	}

	const float get_spl_dis_threshold();
	void calculate_spl_dis_threshold();

	int getWrongNum(){
		int size = spl_positives.size();
		int sum = 0;
		for(int i=0; i<size; ++i)
			if(this->spl_positives[i].second.getTrueLabel() != this->label)
				sum ++;
		return sum;
	}
	void pop_al_positives(const int index) {
		const int size = this->al_positives.size();
		//cout << "index in function pop_al_positives: " << index << endl;
		for (int i = 0; i < size; ++i) {
			//cout << this->al_positives[i].first << endl;
			if (this->al_positives[i].first == index) {
				cout << this->al_positives[i].second.getName() << "is removed from al_positives" << endl;
				this->al_positives[i] = this->al_positives.back();
				this->al_positives.pop_back();
				break;
			}
		}
		for (int i = 0; i < size; ++i) {
			//cout << this->al_incorrect_times[i].first << endl;
			if (this->al_incorrect_times[i].first == index) {
				this->al_incorrect_times[i] = this->al_incorrect_times.back();
				this->al_incorrect_times.pop_back();
				break;
			}
		}
		//cout << "do nothing... " << endl;
		//vector<Index_Photo>::iterator it = this->al_positives.f
	}

	const int get_al_incorrect_times(const int index) {
		for (size_t i = 0; i < this->al_incorrect_times.size(); ++i) {
			if (this->al_incorrect_times[i].first == index) {
				return this->al_incorrect_times[i].second;
			}
		}
		cout << "error: not in al_incorrect_times vector in function get_al_incorrect_times" << endl;
		return 0;
	}

	void add_al_incorrect_time(const int index) {
		for (size_t i = 0; i < this->al_incorrect_times.size(); ++i) {
			if (this->al_incorrect_times[i].first == index) {
				this->al_incorrect_times[i].second++;
				return;
			}
		}
		cout << "error: not in al_incorrect_times vector in function add_al_incorrect_time" << endl;
	}

	void remove_spl_low_dis_index(vector<int>& result);

public:
	bool above_average;
	int current_po_size;
	int current_al_po_size;

private:
	int label;
//	CvSVM* classifier;
//	CvSVMParams params;
	vector<Index_Photo> al_positives;
	vector<Index_Photo> spl_positives;
	vector<AL_Wrong_Index> al_incorrect_times;
	vector<float> spl_weight ;
	model* model_ ;
	float spl_dis_threshold ;
	float avgAccuracy ;
	float spl_dis_threshold_decrease_rate ;

	bool initialization ;
	bool trained ;
};

#endif /* PERSONCLASSIFIER_H_ */
