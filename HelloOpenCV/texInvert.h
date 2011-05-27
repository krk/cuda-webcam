#pragma once

#include "cudaCommon.h"

void deviceTexInvertLaunch(
	float *d_Image,
	int width,
	int height
	);