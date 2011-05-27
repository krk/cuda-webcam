#pragma once

#include "cudaCommon.h"

void deviceTileFlipLaunch(
	float *d_Image,
	int width,
	int height
	);