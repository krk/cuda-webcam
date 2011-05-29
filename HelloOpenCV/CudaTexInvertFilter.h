#ifndef CUDATEXINVERTFILTER_H_
#define CUDATEXINVERTFILTER_H_

/** 
	\file CudaTexInvertFilter.h
	CudaTexInvertFilter sýnýfýnýn tanýmýný içeren baþlýk dosyasý.
*/

#include "common.h"

#include "SingleCudaTexFilter.h"

#include "texInvert.h"

/**
	CUDA ve texture kullanarak GPU üzerinde görüntünün negatifini alan filtre sýnýfý.
*/

class CudaTexInvertFilter : public SingleCudaTexFilter
{

public:

	/**
		\ref deviceTexInvertLaunch metod iþaretçisi ve "texInvert1" sembol adý parametresi ile SingleCudaTexFilter yaratýcýsýný çaðýrýr.
	*/
	CudaTexInvertFilter()
		: SingleCudaTexFilter(deviceTexInvertLaunch, "texInvert1")
	{
	}

private:

	DISALLOW_COPY_AND_ASSIGN(CudaTexInvertFilter);

};

#endif // CUDATEXINVERTFILTER_H_