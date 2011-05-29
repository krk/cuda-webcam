#ifndef CUDATILEFLIPFILTER_H_
#define CUDATILEFLIPFILTER_H_

/** 
	\file CudaTileFlipFilter.h
	CudaTileFlipFilter sýnýfýnýn tanýmýný içeren baþlýk dosyasý.
*/

#include "common.h"

#include "SingleCudaFilter.h"

#include "tileFlip.h" // kernel definition.

/**
	CUDA kullanarak GPU üzerinde görüntüyü blok blok döndüren filtre sýnýfý.
*/

class CudaTileFlipFilter : public SingleCudaFilter
{

public:

	/**
		\ref deviceTileFlipLaunch metod iþaretçisi parametresi ile SingleCudaFilter yaratýcýsýný çaðýrýr.
	*/
	CudaTileFlipFilter()
		: SingleCudaFilter(deviceTileFlipLaunch)
	{
	}

private:

	DISALLOW_COPY_AND_ASSIGN(CudaTileFlipFilter);

};

#endif // CUDATILEFLIPFILTER_H_