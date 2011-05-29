#ifndef TEXBOXBLUR_H_
#define TEXBOXBLUR_H_

/**
	\file texBoxBlur.h
	CUDA texture box blur kernelinin launcher metodunu tanýmlar.
*/

#include "cudaCommon.h"

void deviceTexBoxBlurLaunch(
	float *d_Image,
	int width,
	int height
	);

#endif // TEXBOXBLUR_H_