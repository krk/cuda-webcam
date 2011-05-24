#include "SingleImageProcessor.h"

#pragma once
#define BLOCK_SIZE (16)

inline
unsigned char satchar(float val);

class SingleCudaProcessor : public SingleImageProcessor
{
private:
	float* h_Image;
	float* d_Image;

	// Kernel launcher metodu, device pointer, width ve height'a ihtiyaç duyar.
	void (*kernelLauncher)(float*, int, int);

public:

	SingleCudaProcessor( void kernelLauncher(float*, int, int) )
		: kernelLauncher(kernelLauncher)
	{
	}

	virtual void InitProcessing(int width, int height)
	{
		SingleImageProcessor::InitProcessing(width, height);

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

	virtual void ProcessImage(char* imageData)
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
		checkCUDAError("ProcessImage: memcpy");

		kernelLauncher( d_Image, width, height );
	
		// copy results back to h_C.
		cudaMemcpy( h_Image, d_Image, 3 * sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		checkCUDAError("ProcessImage: memcpy2");

		for(int i=0; i<3*width*height; i++)
		{
			*(imageData + i) = satchar(*(h_Image + i) * 255);
		}
	}

	virtual void ReleaseProcessing()
	{
		SingleImageProcessor::ReleaseProcessing();

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
