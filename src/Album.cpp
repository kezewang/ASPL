#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include "Album.h"
#include "Utils.h"
using namespace cv;
using namespace std;

// load features.txt file
void Album::getFeatures(string fea_path, vector<SparseMat>& features, int fea_dim){
	cout << "loading features... " << endl ;
	ifstream fea_file ;
	fea_file.open( fea_path.c_str(), ios::binary | ios::in ) ;

	int smpNum ;
	fea_file.read( (char*)&smpNum, sizeof( int ) ) ;
	cout << fea_path << endl << "Total SmpNum " << smpNum << endl ;

	int* indexPtr = new int[fea_dim] ;
	float* dataPtr = new float[fea_dim] ;
	for ( int i = 0; i < smpNum; i++ )
	{
		int nonzeroNum ;
		fea_file.read( (char*)&nonzeroNum, sizeof( int ) ) ;

		if ( 0 == nonzeroNum )
			continue ;

		//cout << nonzeroNum << endl;
		fea_file.read( (char*)indexPtr, sizeof( int ) * nonzeroNum ) ;
		fea_file.read( (char*)dataPtr, sizeof( float ) * nonzeroNum ) ;

		int dim_size[] = {fea_dim} ;
		SparseMat tmp_fea( 1, dim_size, CV_32F ) ;

		//cout << nonzeroNum << endl;
		for ( int j = 0; j < nonzeroNum; j++ )
		{
			tmp_fea.ref<float>( indexPtr[j] ) = dataPtr[j] ;
			// if (i == 0) {
			// 	cout << dataPtr[j] << endl;
			// }
		}
		features.push_back(tmp_fea);
	}

	cout << "Loaded SmpNum is " << features.size() << endl ;
	delete [] dataPtr ;
	delete [] indexPtr ;
	fea_file.close() ;
	cout << "Loading feature finished!" << endl ;
}


// load filenames.txt file
void Album::getLabels(string id_path, vector<int>& ids, vector<string>& names){
	//cout << "Collecting labels...   ";
	fstream idFile;
	idFile.open(id_path.c_str());

	if ( !idFile.is_open() )
	{
		cout << "Label file does not exist! & Quit!" << endl ;
		exit( 0 ) ;
	}

	int maxLabel = 0 ;
	while (!idFile.eof())
	{
		string tmp_str;
		int tmp_id = 0;
		idFile >> tmp_str >> tmp_id;

		names.push_back(tmp_str);

		if ( maxLabel < tmp_id )
			maxLabel = tmp_id ;

		ids.push_back(tmp_id);
	}

	maxLabel += 1 ;
	int* existFlags = new int[maxLabel] ;
	for ( int i = 0; i < maxLabel; i++ )
	{
		existFlags[i] = 0 ;
	}
	for ( size_t i = 0; i < ids.size(); i++ )
	{
		existFlags[ids[i]] = 1 ;
	}
	
	int labelCount = 0 ;
	for ( int i = 0; i < maxLabel; i++ )
		labelCount += existFlags[i] ;

	if ( labelCount != maxLabel )
		cout << "Lack of Label: " ;
	for ( int i = 0; i < maxLabel; i++ )
	{
		if ( existFlags[i] == 0 )
			cout << i << "-th " ;
	}
	if ( labelCount != maxLabel )
		cout << "category!" << endl ;
	delete [] existFlags ;

	//cout << "Finished! with label number " << labelCount << " and " << ids.size() - 1 << " samples" << endl;
	idFile.close() ;
}

void Album::reLoadLabels( Config* config )
{
	cout << "Reloading training annotations ..." << endl ;

	vector<int> ids;
	vector<string> names;
	getLabels(config->get_train_id_path(), ids, names);

	for (int photo_index = 0; photo_index < (int)ids.size() - 1; ++photo_index)
	{
		if ( 0 == strcmp( names[photo_index].c_str(), this->database[photo_index].second.getName().c_str() ) )
		{
			this->database[photo_index].second.setTrueLabel(ids[photo_index]);
		}
		else
			cout << names[photo_index] << "reload failed!" << endl ;
	}

	vector<int> init_ids;
	vector<string> init_names;
	getLabels(config->get_init_id_path(), init_ids, init_names);

	int offset = ids.size() - 1;
	for (int photo_index = 0; photo_index < (int)init_ids.size() - 1; ++photo_index)
	{
		if ( offset + photo_index == this->database[offset + photo_index].first && 0 == strcmp( init_names[photo_index].c_str(), this->database[offset + photo_index].second.getName().c_str() ) )
		{
			this->database[offset + photo_index].second.setTrueLabel(init_ids[photo_index]);
		}
		else
			cout << names[photo_index] << "reload failed!" << endl ;
	}

	cout << "Reloading training annotations done!" << endl << endl;
}


void Album::reLoadPhotos(Config* config){
	//cout << "reloading database ..." <<endl;

	vector<SparseMat> features;
	getFeatures(config->get_train_fea_path(), features, config->get_fea_dim());

	int size = features.size();
	for(int i=0; i<size; ++i)
		this->database[i].second.setFeature(features[i]);

	vector<SparseMat> init_features;
	getFeatures(config->get_init_fea_path(), init_features, config->get_fea_dim());

	for(int i=0; i< (int) init_features.size(); ++i){
		this->database[i + size].second.setFeature(init_features[i]);
	}

	//cout << "reload database done." << size << " " << init_features.size() <<endl;
}

void Album::reLoadValidationSet(Config* config){
	//cout << "reloading validation set ..." <<endl;

	vector<SparseMat> features;
	getFeatures(config->get_test_fea_path(), features, config->get_fea_dim());

	vector<int> vali_ids;
	vector<string> vali_names;
	getLabels(config->get_test_id_path(), vali_ids, vali_names);

	map<int, vector<Photo> >::iterator it;

	int size = vali_ids.size() - 1;
	for (it = this->validation_set.begin(); it != this->validation_set.end(); ++it){
		vector<Photo> tv;
		it->second = tv;
	}

	for (int photo_index = 0; photo_index < size; ++photo_index){
		Photo tmp_photo(config);
		tmp_photo.setName(vali_names[photo_index]);
		tmp_photo.setTrueLabel(vali_ids[photo_index]);
		tmp_photo.setFeature(features[photo_index]);

		it = this->validation_set.find(vali_ids[photo_index]);
		it->second.push_back(tmp_photo);
	}

	//cout << "reload validation set done." <<endl <<endl;
}


// load photos
void Album::loadPhotos(Config* config){
	cout << "loading database ... "<<endl;

	vector<SparseMat> features;
	getFeatures(config->get_train_fea_path(), features, config->get_fea_dim());

	vector<SparseMat> init_features;
	getFeatures(config->get_init_fea_path(), init_features, config->get_fea_dim());

	vector<SparseMat> vali_features;
	getFeatures(config->get_test_fea_path(), vali_features, config->get_fea_dim());

	vector<int> ids;
	vector<string> names;
	getLabels(config->get_train_id_path(), ids, names);

	for (int photo_index = 0; photo_index < (int)ids.size() - 1; ++photo_index)
	{
		Photo tmp_photo(config);
		tmp_photo.setName(names[photo_index]);
		tmp_photo.setTrueLabel(ids[photo_index]);
//		tmp_photo.setAssignLabel(ids[photo_index]);
		tmp_photo.setFeature(features[photo_index]);

		this->database.push_back(make_pair(photo_index, tmp_photo));
		this->database_selected.push_back(PHOTO_NOT_SELECTED);
		this->database_unclear.push_back(PHOTO_CLEAR);
		this->database_spl_selected_times.push_back(0);
	}

	vector<int> init_ids;
	vector<string> init_names;
	getLabels(config->get_init_id_path(), init_ids, init_names);

	int data_size = ids.size() - 1;
	for (int photo_index = 0; photo_index < (int)init_ids.size() - 1; ++photo_index){
		Photo tmp_photo(config);
		tmp_photo.setName(init_names[photo_index]);
		tmp_photo.setTrueLabel(init_ids[photo_index]);
		tmp_photo.setAssignLabel(init_ids[photo_index]);
		tmp_photo.setFeature(init_features[photo_index]);
		//cout << tmp_photo.getName() << endl;
		this->database.push_back(make_pair(photo_index + data_size, tmp_photo));
		this->database_selected.push_back(PHOTO_SELECTED);
		this->database_unclear.push_back(PHOTO_CLEAR);
		this->database_spl_selected_times.push_back(0);
	}

	vector<int> vali_ids;
	vector<string> vali_names;
	getLabels(config->get_test_id_path(), vali_ids, vali_names);

	for (int photo_index = 0; photo_index < (int)vali_ids.size() - 1; ++photo_index)
	{
		Photo tmp_photo(config);
		tmp_photo.setName(vali_names[photo_index]);
		tmp_photo.setTrueLabel(vali_ids[photo_index]);
		tmp_photo.setFeature(vali_features[photo_index]);

		map<int, vector<Photo> >::iterator it = this->validation_set.find(vali_ids[photo_index]);
		if(it == this->validation_set.end())
		{
			vector<Photo> photos;
			photos.push_back(tmp_photo);
			this->validation_set.insert(pair<int, vector<Photo> >(vali_ids[photo_index], photos));
		}
		else
			it->second.push_back(tmp_photo);
	}
	cout << "finished!" << endl;
}


int Album::getSelectedNum(){
	int num = 0;
	int size = this->database_selected.size();
	for(int i=0; i<size; ++i)
		if(this->database_selected[i] == PHOTO_SELECTED)
			num ++;
	return num;
}

int Album::getUnclearNum(){
	int num = 0;
	int size = this->database_unclear.size();
	for(int i=0; i<size; ++i)
		if(this->database_unclear[i] == PHOTO_UNCLEAR)
			num ++;
	return num;
}

bool pair_cmp(Index_Dis i, Index_Dis j){
	return (i.second < j.second);
}

int Album::getQueriedNum(){
	int num = 0;
	int size = this->database_selected.size();
	for(int i=0; i<size; ++i)
		if(this->database[i].second.isQueryLabel)
			num ++;
	return num;
}


void Album::getDisSortedIndex(Photo& photo, vector<int>& sortedInx, Config* config){
	int size = this->database.size();
	vector<Index_Dis> dis;

	const float* seed_fea = photo.getFeature().ptr<float>(0);
	for(int i=0; i<size; ++i){
		if(this->database_selected[i] == PHOTO_NOT_SELECTED)
			dis.push_back(make_pair(i, distance(seed_fea, this->database[i].second.getFeature().ptr<float>(0), config->get_fea_dim())));
	}

	sort(dis.begin(), dis.end(), pair_cmp);

	size = dis.size();

	for(int i=0; i<size; ++i){
		sortedInx.push_back(dis[i].first);
	}
}

void Album::getNotSelectedIndex(vector<int>& index, const int flag){
	// if flag==0 return set is equal to test
	// if flag==1 return set not include test
	vector<int> temp;
	int size = this->database_selected.size();
	for(int i=0; i<size; ++i)
		if(this->database_selected[i] == PHOTO_NOT_SELECTED && this->database[i].second.isQueryLabel == false)
			temp.push_back(i);

	vector<int> random_index;
	getRandomList(temp.size(), random_index);
	for (size_t i = 0; i < temp.size(); ++i) {
		index.push_back(temp[random_index[i]]);
	}

}

void Album::getSelectedIndex(vector<int>& index){
	int size = this->database_selected.size();
	for(int i=0; i<size; ++i)
		if(this->database_selected[i] == PHOTO_SELECTED || this->database[i].second.isQueryLabel == true)
			index.push_back(i);
}

void Album::get_spl_selected_times(vector<int>& index){
	int size = this->database_spl_selected_times.size();
	for (int i = 0; i < size; ++i) {
		index.push_back(this->database_spl_selected_times[i]);
	}
}

