#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <vector>
#include <cv.h>
#include "linear.h"

#pragma once

int main_train(int argc, const char **argv, const std::vector<cv::SparseMat>& data, float* weight, float* label ) ;

