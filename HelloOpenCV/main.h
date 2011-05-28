#pragma once

#include <stdio.h>
#include <stdlib.h>

#include "opencv2\opencv.hpp"

#include "cudaCommon.h"

#include "SingleImageFilterChain.h"

#include "SingleCudaFilter.h"
#include "CpuInvertFilter.h"

#include "SingleCudaTexFilter.h"

// CUDA kernel launcherlar

#include "invert.h"
#include "tileFlip.h"

#include "texInvert.h"
#include "texBoxBlur.h"
#include "texAbsDiff.h"

void FilterFrame(IplImage* videoFrame);