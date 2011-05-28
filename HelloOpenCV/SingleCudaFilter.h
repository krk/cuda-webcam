#include "cudaCommon.h"

#include "SingleImageFilter.h"

#pragma once

inline
unsigned char satchar(float val);

class SingleCudaFilter : public SingleImageFilter
{
protected:
	float* h_Image;
	float* d_Image;

	// Kernel launcher metodu, device pointer, width ve height'a ihtiyaç duyar.
	void (*kernelLauncher)(float*, int, int);

public:

	explicit SingleCudaFilter( void kernelLauncher(float*, int, int) )
		: kernelLauncher(kernelLauncher)
	{
	}

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

	virtual void ReleaseFilter()
	{
		SingleImageFilter::ReleaseFilter();

		cudaFree( d_Image );
		checkCUDAError("free device image");
	
		cudaFreeHost( h_Image );
		checkCUDAError("free host image");
	}

};

inline
unsigned char satchar(float val)
{
	if(val<=0)
		return (unsigned char)0;
	if(val>=255)
		return (unsigned char)255;

	return (unsigned char)val;
}
