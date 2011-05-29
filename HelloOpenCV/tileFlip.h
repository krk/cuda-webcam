#ifndef TILEFLIP_H_
#define TILEFLIP_H_

/**
	\file tileFlip.h
	CUDA tile flip kernelinin launcher metodunu tanýmlar.
*/

#include "cudaCommon.h"

void deviceTileFlipLaunch(
	float *d_Image,
	int width,
	int height
	);

#endif // TILEFLIP_H_