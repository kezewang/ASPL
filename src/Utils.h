/*
 * Utils.h
 *
 */

#include <vector>
using namespace std;


#ifndef UTILS_H_
#define UTILS_H_


float distance(const float* fea_1, const float* fea_2, int fea_dim);


void getRandomList(int size, vector<int>& randomList);

#endif /* UTILS_H_ */
