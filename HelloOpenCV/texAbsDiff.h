#pragma once

#include "cudaCommon.h"

void deviceTexAbsDiffLaunch(
	float *d_Image,
	int width,
	int height
	);