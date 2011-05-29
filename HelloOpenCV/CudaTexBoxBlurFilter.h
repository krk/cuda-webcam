#ifndef CUDATEXBOXBLURFILTER_H_
#define CUDATEXBOXBLURFILTER_H_

/** 
	\file CudaTexBoxBlurFilter.h
	CudaTexBoxBlurFilter sýnýfýnýn tanýmýný içeren baþlýk dosyasý.
*/

#include "common.h"

#include "SingleCudaTexFilter.h"

#include "texBoxBlur.h" // kernel definition

/**
	CUDA ve texture kullanarak GPU üzerinde görüntüyü bulandýran filtre sýnýfý.
*/

class CudaTexBoxBlurFilter : public SingleCudaTexFilter
{

public:

	/**
		\ref deviceTexBoxBlurLaunch metod iþaretçisi ve "texBlur1" sembol adý parametresi ile SingleCudaTexFilter yaratýcýsýný çaðýrýr.
	*/
	CudaTexBoxBlurFilter()
		: SingleCudaTexFilter(deviceTexBoxBlurLaunch, "texBlur1")
	{
	}

private:

	DISALLOW_COPY_AND_ASSIGN(CudaTexBoxBlurFilter);

};

#endif // CUDATEXBOXBLURFILTER_H_