#include "config.h"
#include <fstream>
using namespace std;

const int DEFAULT_FEA_DIM = 400;

void Config::init(){

}


set_Field Config::get_Field_Func(string& field_name){
	set_Field func = NULL;
	if(field_name == "train_id_path")
		func = &Config::set_train_id_path;
	else if(field_name == "train_fea_path")
		func = &Config::set_train_fea_path;
	else if(field_name == "init_id_path")
		func = &Config::set_init_id_path;
	else if(field_name == "init_fea_path")
		func = &Config::set_init_fea_path;
	else if(field_name == "test_id_path")
		func = &Config::set_test_id_path;
	else if(field_name == "test_fea_path")
		func = &Config::set_test_fea_path;
	else if(field_name == "finetune_id_path")
		func = &Config::set_finetune_id_path;
	else if(field_name == "finetune_script_path")
		func = &Config::set_finetune_script_path;
	else if(field_name == "get_features_script_path")
		func = &Config::set_features_script_path;
	else if(field_name == "fea_dim")
		func = &Config::set_fea_dim;
	else if(field_name == "al_select_per_iter")
		func = &Config::set_al_select_per_iter;
	else if(field_name == "all_select_num")
		func = &Config::set_all_select_num;
	return func;
}


void Config::readConfig(char* config_file_path){
	fstream config_File;
	config_File.open(config_file_path);

	string line;
	set_Field func;
	string temp_field;
	int index;

	while (!config_File.eof()){
		config_File >> line;
		if(line[0] != '#'){
			index = line.find_last_of(':');
			temp_field = line.substr(0, index);
			func = get_Field_Func(temp_field);
			(this->*func)(line.substr(index + 1, line.length()).c_str());
		}
	}
}




