#ifndef TEXABSDIFF_H_
#define TEXABSDIFF_H_

/**
	\file texAbsDiff.h
	CUDA texture absolute difference kernelinin launcher metodunu tanýmlar.
*/

#include "cudaCommon.h"

void deviceTexAbsDiffLaunch(
	float *d_Image,
	int width,
	int height
	);

#endif // TEXABSDIFF_H_