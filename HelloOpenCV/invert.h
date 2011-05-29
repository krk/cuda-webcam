#ifndef INVERT_H_
#define INVERT_H_

/**
	\file invert.h
	CUDA invert kernelinin launcher metodunu tanýmlar.
*/

#include "cudaCommon.h"

void deviceInvertLaunch(
	float *d_Image,
	int width,
	int height
	);

#endif // INVERT_H_