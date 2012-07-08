#include "FilterFactory.h"

#include "common.h"

#include "CpuCCLFilter.h"
#include "CpuInvertFilter.h"
#include "CpuMovingAverageFilter.h"
#include "CudaInvertFilter.h"
#include "CudaSepiaFilter.h"
#include "CudaTexBoxBlurFilter.h"
#include "CudaTexInvertFilter.h"
#include "CudaTileFlipFilter.h"
#include "IdentityFilter.h"
#include "ThresholdFilter.h"

#include "cudaCommon.h"

/** 
	\file FilterFactory.cpp
	CudaFilters projesindeki filtreler için factory metodlarýný içeren dosya.
*/

FILTERAPI void FILTERENTRY ReleaseCUDAThread()
{
	cudaThreadExit();
}

FILTERAPI ISingleImageFilter* FILTERENTRY GetCpuCCLFilter()
{
	return new CpuCCLFilter();	
}

FILTERAPI ISingleImageFilter* FILTERENTRY GetCpuInvertFilter()
{
	return new CpuInvertFilter();
}

FILTERAPI ISingleImageFilter* FILTERENTRY GetCpuMovingAverageFilter( int framesToAverage )
{
	return new CpuMovingAverageFilter( framesToAverage );
}

FILTERAPI ISingleImageFilter* FILTERENTRY GetCudaInvertFilter()
{
	return new CudaInvertFilter();
}

FILTERAPI ISingleImageFilter* FILTERENTRY GetCudaSepiaFilter()
{
	return new CudaSepiaFilter();
}

FILTERAPI ISingleImageFilter* FILTERENTRY GetCudaTexBoxBlurFilter()
{
	return new CudaTexBoxBlurFilter();
}

FILTERAPI ISingleImageFilter* FILTERENTRY GetCudaTexInvertFilter()
{
	return new CudaTexInvertFilter();
}

FILTERAPI ISingleImageFilter* FILTERENTRY GetCudaTileFlipFilter()
{
	return new CudaTileFlipFilter();
}

FILTERAPI ISingleImageFilter* FILTERENTRY GetIdentityFilter()
{
	return new IdentityFilter();
}

FILTERAPI ISingleImageFilter* FILTERENTRY GetThresholdFilter( unsigned char threshold )
{
	return new ThresholdFilter( threshold );
}