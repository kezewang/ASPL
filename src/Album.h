#include <map>
#include <list>
#include "Photo.hpp"

#ifndef ALBUM_H_
#define ALBUM_H_

typedef pair<int, float> Index_Dis;

bool pair_cmp(Index_Dis i, Index_Dis j);

const int PHOTO_CLEAR = 3 ;
const int PHOTO_UNCLEAR = 2 ;
const int PHOTO_SELECTED 	  = 1;
const int PHOTO_NOT_SELECTED  = 0;
const int PHOTO_SPL_SELECTED_ONCE = 1;
const int PHOTO_SPL_SELECTED_TWICE = 2;

typedef pair<int, Photo> Index_Photo;

class Album{
public:
	Album(){this->iter = 0;}
	virtual ~Album(){}

	//void loadInitSamples()
	// load photos
	void loadPhotos(Config* config);

	void reLoadPhotos(Config* config);

	void reLoadLabels( Config* config ) ;

	void reLoadValidationSet(Config* config);

	void setSelected(int index){
		int size = database_selected.size();
		assert(index < size && index >= 0);
		this->database_selected[index] = PHOTO_SELECTED;
	}

	void setUnclear(int index){
		int size = database_selected.size();
		assert(index < size && index >= 0);
		this->database_unclear[index] = PHOTO_UNCLEAR;
	}

	void setClear(int index){
		int size = database_selected.size();
		assert(index < size && index >= 0);
		this->database_unclear[index] = PHOTO_CLEAR;
	}

	void setNotSelected(int index){
		int size = database_selected.size();
		assert(index < size && index >= 0);
		this->database_selected[index] = PHOTO_NOT_SELECTED;
	}

	bool isSelected(int index){
		int size = database_selected.size();
		assert(index < size && index >= 0);
		return this->database_selected[index] == PHOTO_SELECTED;
	}

	bool isUnclear(int index){
		int size = database_selected.size();
		assert(index < size && index >= 0);
		return this->database_unclear[index] == PHOTO_UNCLEAR;
	}

	void getDisSortedIndex(Photo& photo, vector<int>& sortedInx, Config* config);
	int getSelectedNum();
	int getUnclearNum() ;
	int getQueriedNum();
	void getNotSelectedIndex(vector<int>& index, const int flag = 0);
	void getSelectedIndex(vector<int>& index);
	int get_spl_selected_times(int index)
	{
		int size = this->database_spl_selected_times.size();
		assert(index < size && index >= 0);
		return this->database_spl_selected_times[index];
	}
	void spl_selected_times_update_once(int index)
	{
		int size = this->database_spl_selected_times.size();
		assert(index < size && index >= 0);
		this->database_spl_selected_times[index]++;
		/*if (this->database_spl_selected_times[index] == 0)
			this->database_spl_selected_times[index] = PHOTO_SPL_SELECTED_ONCE;
		else
			this->database_spl_selected_times[index] = PHOTO_SPL_SELECTED_TWICE;*/
	}
	bool is_spl_always_selected(int index)
	{
		int size = this->database_spl_selected_times.size();
		assert(index < size && index >= 0);
		return this->database_spl_selected_times[index]==PHOTO_SPL_SELECTED_TWICE;
	}
	int get_spl_selected_vector_size() {return this->database_spl_selected_times.size();}
	void get_spl_selected_times(vector<int>& index);
	void set_iter(int iter) {this->iter = iter;}
private:
	// load features.txt file
//	void getFeatures(string fea_path, vector<Mat>& features, int fea_dim = 400);
	void getFeatures(string fea_path, vector<SparseMat>& features, int fea_dim = 400);

	// load filenames.txt file
	void getLabels(string id_path, vector<int>& ids, vector<string>& names);

public:
	//vector<Index_Photo> initSamples;
	vector<Index_Photo> database;
	map<int, vector<Photo> > validation_set;

private:
	vector<int> database_selected;
	vector<int> database_unclear;
	vector<int> database_spl_selected_times;
	int iter;
};

#endif /* ALBUM_H_ */
