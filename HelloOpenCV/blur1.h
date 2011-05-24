#pragma once

#include "cudaCommon.h"

void deviceBlur1Launch(
	float *d_Image,
	int width,
	int height
	);