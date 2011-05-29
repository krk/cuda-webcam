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
	virtual void InitFilter(int width, int height)
	{
		SingleImageFilter::InitFilter(width, height);

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
		// copy imageData to GPU.
		for(int i=0; i<3*width*height; i++)
		{
			*(h_Image + i) = (unsigned char)*(imageData + i) / 255.0f; // normalize and copy image
		}

		/*
		Copy image to device.
		*/

		cudaMemcpy( d_Image, h_Image, 3 * sizeof(float) * width * height, cudaMemcpyHostToDevice );
		checkCUDAError("FilterImage: memcpy");

		kernelLauncher( d_Image, width, height );
	
		// copy results back to h_C.
		cudaMemcpy( h_Image, d_Image, 3 * sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		checkCUDAError("FilterImage: memcpy2");

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