#pragma once

#include "cudaCommon.h"

void deviceTexBoxBlurLaunch(
	float *d_Image,
	int width,
	int height
	);