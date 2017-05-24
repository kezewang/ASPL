/*
 * config.h
 *
 *  read configure file and get config paramaters
 */

#include<string>
using namespace std;
#include<stdlib.h>


#ifndef CONFIG_H_
#define CONFIG_H_


class Config;

typedef void (Config::*set_Field)(const char*);

/*
  	 train_id_path: the path to read the <path, label> for the training samples
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

class Config{
public:
	Config(char* config_file_path){
		init();
		this->readConfig(config_file_path);
	};

	~Config(){}

private:
	void readConfig(char* config_file_path);
	void init();
	set_Field get_Field_Func(string& field_name);

public:
	const string get_train_id_path(){
		return this->train_id_path;
	}

	const string get_train_fea_path(){
		return this->train_fea_path;
	}

	const string get_init_id_path(){
		return this->init_id_path;
	}

	const string get_init_fea_path(){
		return this->init_fea_path;
	}

	const string get_test_id_path(){
		return this->test_id_path;
	}

	const string get_test_fea_path(){
		return this->test_fea_path;
	}

	const string get_finetune_id_path(){
		return this->finetune_id_path;
	}

	const string get_finetune_script_path(){
		return this->finetune_script_path;
	}

	const string get_features_script_path(){
		return this->features_script_path;
	}

	const int get_fea_dim(){
		return this->fea_dim;
	}

	const int get_al_select_per_iter(){
		return this->al_select_per_iter;
	}

	const int get_all_select_num(){
		return this->all_select_num;
	}

	const int get_al_random_select_num(){
		return this->al_random_select_num;
	}

	const float get_al_diversity_threshold(){
		return this->al_diversity_threshold;
	}

	const int get_al_diversity_select_num(){
		return this->al_diversity_select_num;
	}

	const float get_al_spl_verification_dis_threshold(){
		return this->al_spl_verification_dis_threshold;
	}

	void update_all_select_num(const int iter) {
		this->all_select_num = 10000 + iter * iter * 10;
	}

private:

	void set_train_id_path(const char* value){
		this->train_id_path = *(new string(value));
	}

	void set_train_fea_path(const char* value){
		this->train_fea_path = *(new string(value));
	}

	void set_init_id_path(const char* value){
		this->init_id_path = *(new string(value));
	}

	void set_init_fea_path(const char* value){
		this->init_fea_path = *(new string(value));
	}

	void set_test_id_path(const char* value){
		this->test_id_path = *(new string(value));
	}

	void set_test_fea_path(const char* value){
		this->test_fea_path = *(new string(value));
	}

	void set_finetune_id_path(const char* value){
		this->finetune_id_path = *(new string(value));
	}

	void set_finetune_script_path(const char* value){
		this->finetune_script_path = *(new string(value));
	}

	void set_features_script_path(const char* value){
		this->features_script_path = *(new string(value));
	}

	void set_fea_dim(const char* value){
		this->fea_dim = atoi(value);
	}

	void set_al_select_per_iter(const char* value){
		this->al_select_per_iter = atoi(value);
	}

	void set_all_select_num(const char* value){
		this->all_select_num = atoi(value);
	}


private:
	string train_id_path;
	string train_fea_path;
	string init_id_path;
	string init_fea_path;
	string test_id_path;
	string test_fea_path;
	string finetune_id_path;
	string finetune_script_path;
	string features_script_path;
	int fea_dim;
	int al_select_per_iter;
	int all_select_num;
	int al_diversity_select_num ;
	int al_random_select_num ;

	float al_diversity_threshold ;
	float al_spl_verification_dis_threshold ;
};


#endif /* CONFIG_H_ */
