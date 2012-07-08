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

#ifndef SINGLECUDAFILTER_H_
#define SINGLECUDAFILTER_H_

/** 
	\file SingleCudaFilter.h
	SingleCudaFilter sýnýfýnýn tanýmýný içeren baþlýk dosyasý.
*/

#include "common.h"

#include "cudaCommon.h"

#include "SingleImageFilter.h"

/** Görüntü iþleme kernellerinin ihtiyacý olan görüntünün GPU üzerindeki adresi, piksel geniþliði ve piksel yüksekliðini parametre olarak alan metod iþaretçisi. */
typedef void (*ptKernelLauncher)(float*, int, int);

inline
unsigned char satchar(float val);

/**
	CUDA kullanarak GPU üzerinde görüntü filtrelemek için kullanýlan sýnýf.
	
	SingleCudaFilter sýnýfýnýn görevi, FilterImage metoduna gönderilen görüntüyü normalize ederek GPU belleðine aktarmak ve bu belleði yönetmektir.
*/

class SingleCudaFilter : public SingleImageFilter
{
protected:
	float* h_Image; /**< Normalize edilmiþ görüntünün CPU bellek adresi. */
	float* d_Image; /**< Normalize edilmiþ görüntünün GPU bellek adresi. */

	ptKernelLauncher kernelLauncher;

	DISALLOW_COPY_AND_ASSIGN(SingleCudaFilter);

public:

	/**
		kernelLauncher metod iþaretçisini alan yaratýcý.

		\param kernelLauncher \ref kernelLauncher tipinde metod iþaretçisi alan yaratýcý.
	*/
	explicit SingleCudaFilter( ptKernelLauncher kernelLauncher )
		: kernelLauncher(kernelLauncher)
	{
	}

	/**
		CPU ve GPU üzerinde normalize edilmiþ görüntüler için bellek ayýrýr.
	*/
	virtual void InitFilter(int width, int height, int rowStride)
	{
		SingleImageFilter::InitFilter(width, height, rowStride);

		/*
		allocate device memory
		*/

		cudaMalloc( (void**) &d_Image, 3 * sizeof(float) * width * height );
		checkCUDAError("malloc device image");


		/*
		allocate host memory
		*/

		cudaMallocHost( (void**) &h_Image, 3 * sizeof(float) * width * height );
		checkCUDAError("malloc host image");
	}

	/**
		Yaratýcýda alýnan kerneli çaðýrýr.

		\param imageData Görüntünün BGR kanal sýralý bellekteki adresi.


		Görüntüyü normalize ederek kernelLauncher iþaretçisinin gösterdiði metodu çaðýrýr ardýndan görüntüyü denormalize eder( [0, 255] aralýðýna ).
		Kernelde iþlenen görüntüden sonuç olarak [0, 1] aralýðý dýþýnda bir deðer dönerse o kanalýn deðeri [0, 255] aralýðýndan dýþarýda olabilir. Bu durumda deðer yakýn olduðu sýnýra indirgenir.
	*/
	virtual void FilterImage(char* imageData)
	{
		// imageData deðiþkenindeki görüntü verisi normalize edilerek h_Image deðiþkenine aktarýlýr.
		for(int i=0; i<3*width*height; i++)
		{
			*(h_Image + i) = (unsigned char)*(imageData + i) / 255.0f; // normalize and copy image
		}

		/*
			Görüntü GPU belleðine kopyalanýr.
		*/

		cudaMemcpy( d_Image, h_Image, 3 * sizeof(float) * width * height, cudaMemcpyHostToDevice );
		checkCUDAError("FilterImage: memcpy");

		/*
			Constructorda verilen kernel çalýþtýrýlýr.
		*/
		kernelLauncher( d_Image, width, height );
	
		/*
			Sonuçlar CPU belleðine kopyalanýr.
		*/
		cudaMemcpy( h_Image, d_Image, 3 * sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		checkCUDAError("FilterImage: memcpy2");

		/*
			h_Image deðiþkenindeki normalize edilmiþ görüntü verisi [0, 255] aralýðýna çekilir.
		*/
		for(int i=0; i<3*width*height; i++)
		{
			*(imageData + i) = satchar(*(h_Image + i) * 255);
		}
	}

	/**
		CPU ve GPU üzerinde normalize edilmiþ görüntüler için ayrýlmýþ belleði serbest býrakýr.
	*/
	virtual void ReleaseFilter()
	{
		if( this->isReleased )
			return;

		SingleImageFilter::ReleaseFilter();

		cudaFree( d_Image );
		checkCUDAError("free device image");
	
		cudaFreeHost( h_Image );
		checkCUDAError("free host image");
	}

};

/**
	val parametresini [0, 255] aralýðýna indirger.

	\param val Ýndirgenecek parametre.
*/
inline
unsigned char satchar(float val)
{
	if(val<=0)
		return (unsigned char)0;
	if(val>=255)
		return (unsigned char)255;

	return (unsigned char)val;
}


#endif // SINGLECUDAFILTER_H_