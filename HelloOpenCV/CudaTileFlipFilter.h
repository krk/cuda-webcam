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