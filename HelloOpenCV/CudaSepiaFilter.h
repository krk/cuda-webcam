#ifndef CUDASEPUAFILTER_H_
#define CUDASEPUAFILTER_H_

/** 
	\file CudaSepiaFilter.h
	CudaSepiaFilter sýnýfýnýn tanýmýný içeren baþlýk dosyasý.
*/

#include "common.h"

#include "SingleCudaFilter.h"

#include "sepia.h" // kernel definition.

/**
	CUDA kullanarak GPU üzerinde görüntünün sepia tonlamasýný döndüren filtre sýnýfý.
*/

class CudaSepiaFilter : public SingleCudaFilter
{

public:

	/**
		\ref deviceSepiaLaunch metod iþaretçisi parametresi ile SingleCudaFilter yaratýcýsýný çaðýrýr.
	*/
	CudaSepiaFilter()
		: SingleCudaFilter(deviceSepiaLaunch)
	{
	}

private:

	DISALLOW_COPY_AND_ASSIGN(CudaSepiaFilter);

};

#endif // CUDASEPUAFILTER_H_