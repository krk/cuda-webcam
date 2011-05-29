#ifndef MAIN_H_
#define MAIN_H_

/**
	\file main.h
	Main.cpp dosyasýndan içerilen baþlýk dosyasý.
*/

#include <stdio.h>
#include <stdlib.h>

#include "opencv2\opencv.hpp"

#include "common.h"

#include "cudaCommon.h"


#include "SingleImageFilterChain.h"

#include "SingleCudaFilter.h"
#include "CpuInvertFilter.h"

#include "SingleCudaTexFilter.h"

#include "IdentityFilter.h"
// CUDA kernel launcherlar

#include "invert.h"
#include "tileFlip.h"

#include "texInvert.h"
#include "texBoxBlur.h"
#include "texAbsDiff.h"

#include "CudaTileFlipFilter.h"
#include "CudaInvertFilter.h"
#include "CudaTexBoxBlurFilter.h"
#include "CudaTexInvertFilter.h"

#endif // MAIN_H_