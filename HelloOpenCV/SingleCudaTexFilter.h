// Copyright (c) 2011 Kerem KAT 
// 
// http://dissipatedheat.com/
// Do not hesisate to contact me about usage of the code or to make comments 
// about the code. Your feedback will be appreciated.
// keremkat<@>gmail<.>com
//
// Kodun kullanýmý hakkýnda veya yorum yapmak için benimle iletiþim kurmaktan
// çekinmeyiniz. Geri bildirimleriniz deðerlendirilecektir.
// keremkat<@>gmail<.>com
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to 
// deal in the Software without restriction, including without limitation the 
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
// sell copies of the Software, and to permit persons to whom the Software is 
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

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