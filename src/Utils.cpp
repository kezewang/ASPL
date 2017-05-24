/*
 * Utils.cpp
 *
 */

#include <cmath>
#include <ctime>
#include <stdlib.h>
#include "Utils.h"


float distance(const float* fea_1, const float* fea_2, int fea_dim){
	float dis = 0;
	for(int i=0; i<fea_dim; ++i)
		dis += pow(fea_1[i] - fea_2[i], 2);
	return dis;
}


void getRandomList(int size, vector<int>& randomList){
	int *flag = new int[size];
	for(int i=0; i<size; ++i) flag[i] = 0;
	int sum = 0;
	int index;
	srand((unsigned)time(0));
	while(sum < size){
		index = rand() % size;
		if(flag[index] == 0){
			flag[index] = 1;
			sum ++;
			randomList.push_back(index);
		}
	}
	delete[] flag;
}
