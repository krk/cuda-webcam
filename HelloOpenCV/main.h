#pragma once

#include <stdio.h>
#include <stdlib.h>

#include "opencv2\opencv.hpp"

#include "cudaCommon.h"

#include "SingleCudaProcessor.h"
#include "CpuInvertFilter.h"

// CUDA kernel launcherlar

#include "invert.h"
#include "blur1.h"

void ProcessFrame(IplImage* videoFrame);