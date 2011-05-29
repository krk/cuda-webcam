#ifndef TEXINVERT_H_
#define TEXINVERT_H_

/**
	\file texInvert.h
	CUDA texture invert kernelinin launcher metodunu tanýmlar.
*/

#include "cudaCommon.h"

void deviceTexInvertLaunch(
	float *d_Image,
	int width,
	int height
	);

#endif TEXINVERT_H_