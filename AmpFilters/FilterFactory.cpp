#include "FilterFactory.h"

#include "..\CudaFilters\common.h"

#include "AmpInvertFilter.h"

/** 
	\file FilterFactory.cpp
	AmpFilters projesindeki filtreler için factory metodlarýný içeren dosya.
*/

FILTERAPI ISingleImageFilter* FILTERENTRY GetAmpInvertFilter()
{
	return new AmpInvertFilter();
}