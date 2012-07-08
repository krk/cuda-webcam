#include "FilterFactory.h"
#include "CpuInvertFilter.h"
#include "common.h"
#include "CudaInvertFilter.h"

//#include "cudaCommon.h"

FILTERAPI ISingleImageFilter* FILTERENTRY GetCpuInvertFilter()
{
	return new CpuInvertFilter();
}

FILTERAPI ISingleImageFilter* FILTERENTRY GetCudaInvertFilter()
{
	return new CudaInvertFilter();
}

FILTERAPI void FILTERENTRY ReleaseCUDAThread()
{
//	cudaThreadExit();
}