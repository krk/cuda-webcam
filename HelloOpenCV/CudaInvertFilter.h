#ifndef CUDAINVERTFILTER_H_
#define CUDAINVERTFILTER_H_

/** 
	\file CudaInvertFilter.h
	CudaInvertFilter sýnýfýnýn tanýmýný içeren baþlýk dosyasý.
*/

#include "common.h"

#include "SingleCudaFilter.h"

#include "invert.h" // kernel definition

/**
	CUDA kullanarak GPU üzerinde görüntünün negatifini alan filtre sýnýfý.
*/

class CudaInvertFilter : public SingleCudaFilter
{

public:

	/**
		\ref deviceInvertLaunch metod iþaretçisi parametresi ile SingleCudaFilter yaratýcýsýný çaðýrýr.
	*/
	CudaInvertFilter()
		: SingleCudaFilter(deviceInvertLaunch)
	{
	}

private:

	DISALLOW_COPY_AND_ASSIGN(CudaInvertFilter);

};

#endif // CUDAINVERTFILTER_H_