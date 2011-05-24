#pragma once

#include "cudaCommon.h"

inline 
unsigned char satchar(float val);

inline
void deviceInvertLaunch(
	int width,
	int height,
	float *d_Image,
	float* h_Image);