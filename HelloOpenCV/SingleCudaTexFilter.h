#ifndef SINGLECUDATEXFILTER_H_
#define SINGLECUDATEXFILTER_H_

/** 
	\file SingleCudaTexFilter.h
	SingleCudaTexFilter sýnýfýnýn tanýmýný içeren baþlýk dosyasý.
*/

#include "common.h"

#include "cudaCommon.h"

#include "SingleCudaFilter.h"

/**
	CUDA ve texture kullanarak GPU üzerinde görüntü filtrelemek için kullanýlan sýnýf.
	
	SingleCudaTexFilter sýnýfýnýn görevi, SingleCudaFilter sýnýfýna ek olarak CUDA üzerinde texture kullanan kernelleri çaðýrabilmek için texture yüklemesi ve texture yönetimi yapmaktýr.
*/

class SingleCudaTexFilter : public SingleCudaFilter
{

private:
	cudaArray* cu_array;
	cudaChannelFormatDesc channelDesc;
	const char* textureSymbolName;

	const textureReference* constTexRefPtr;
	textureReference* texRefPtr;

	DISALLOW_COPY_AND_ASSIGN(SingleCudaTexFilter);

public:
	
	/**
		kernelLauncher ve texture adýný alan SingleCudaTexFilter yaratýcýsý.

		\param kernelLauncher Kerneli çaðýrmak için kullanýlan metod iþaretçisi.
		\param textureSymbolName Kernelde kullanýlan texture'ýn sembol adý.
		
		ptKernelLauncher tipinde metod iþaretçisi ve texture sembolünün kerneldeki adýný alan SingleCudaTexFilter yaratýcýsý.
	*/
	explicit SingleCudaTexFilter( ptKernelLauncher kernelLauncher, const char* textureSymbolName )
		: SingleCudaFilter(kernelLauncher), 
		textureSymbolName(textureSymbolName),
		constTexRefPtr(NULL),
		texRefPtr(NULL)
	{
	}

	virtual void InitFilter(int width, int height);
	virtual void FilterImage(char* imageData);
	virtual void ReleaseFilter();
};

#endif SINGLECUDATEXFILTER_H_