/*
 * Photo.h
 *
 */
#include<iostream>
#include <string>
#include <vector>
#include <cv.h>
#include "config.h"

using namespace cv;
using namespace std;

#ifndef PHOTO_H_
#define PHOTO_H_

const int LABEL_NOT_ASSIGNED = -1;

class Photo{
public:
//	Photo(){}
	Photo(Config* config);
	virtual ~Photo(){}

	// return the index according to similarity
	//vector<int> retrieveSimilarPhotos(const vector<Photo>& database);

	// returns the square of distance
	float getDistance(Photo& p);

	bool isAssignLabelCorrect();
	bool hasAssingedLabel();

	// getter and setter
	void setTrueLabel(int n);
	const int getTrueLabel();

	void setName(string s);
	const string getName();

	void setAssignLabel(int assign);
	int getAssignedLabel();

	void setFeature(SparseMat tmp_feature);
	void setFeature( Mat tmp_feature ) ;

	const SparseMat getSparseFeature();
	const Mat getFeature();

	void addNegLabel(int neg);
	const vector<int> getNegLabel();

	const int getFeatureDimension() { return fea_dim; }

	bool isQueryLabel;

private:
	string name; // eg : 3_10.jpg  (10th image of person 3)
	int true_label;
	int assigned_label;
	//vector<int> neg_label; // not belong to neg_label
	SparseMat feature;
	int fea_dim;
};

#endif /* PHOTO_H_ */
