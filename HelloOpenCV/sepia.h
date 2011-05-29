#ifndef SEPIA_H_
#define SEPIA_H_

/**
	\file sepia.h
	CUDA sepia kernelinin launcher metodunu tanýmlar.
*/

#include "cudaCommon.h"

void deviceSepiaLaunch(
	float *d_Image,
	int width,
	int height
	);

#endif // SEPIA_H_