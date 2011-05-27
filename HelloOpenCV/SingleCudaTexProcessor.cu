 //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
 //   cudaArray* cu_array;
 //   cutilSafeCall( cudaMallocArray( &cu_array, &channelDesc, width, height )); 
 //   cutilSafeCall( cudaMemcpyToArray( cu_array, 0, 0, h_data, size, cudaMemcpyHostToDevice));

 //   // set texture parameters
 //   tex.addressMode[0] = cudaAddressModeWrap;
 //   tex.addressMode[1] = cudaAddressModeWrap;
 //   tex.filterMode = cudaFilterModeLinear;
 //   tex.normalized = true;    // access with normalized texture coordinates

 //   // Bind the array to the texture
 //   cutilSafeCall( cudaBindTextureToArray( tex, cu_array, channelDesc));

#pragma once


#include "SingleCudaProcessor.h"

#include "cudaCommon.h"

#include "TextureHeader.cu"


class SingleCudaTexProcessor : public SingleCudaProcessor
{
private:
	float* h_Image;
	float* d_Image;

	cudaArray* cu_array;
	cudaChannelFormatDesc channelDesc;

	// Kernel launcher metodu, device pointer, width ve height'a ihtiyaç duyar.
	void (*kernelLauncher)(float*, int, int);

public:

	SingleCudaTexProcessor( void kernelLauncher(float*, int, int) )
		: SingleCudaProcessor(kernelLauncher)
	{
	}

	virtual void InitProcessing(int width, int height)
	{
		SingleCudaTexProcessor::InitProcessing(width, height);

		/*
		allocate device texture memory
		*/

		channelDesc = cudaCreateChannelDesc<float4>();
		cudaMallocArray( &cu_array, &channelDesc, width, height ); 
		checkCUDAError("malloc device image");
 
		/*
		allocate device memory for result.
		*/

		cudaMalloc( (void**) &d_Image, 3 * sizeof(float) * width * height );
		checkCUDAError("malloc device image2");

		/*
		allocate host memory
		*/

		cudaMallocHost( (void**) &h_Image, 4 * sizeof(float) * width * height );
		checkCUDAError("malloc host image");
	}

	virtual void ProcessImage(char* imageData)
	{
		int index;
		// copy imageData to GPU.
		for(int i=0; i<4*width*height; i+=4)
		{
			index = (i/4) * 3; // 4.kanal boþ, float4 için, kernellerde kullanýlmaz.
			*(h_Image + i) = (unsigned char)*(imageData + index) / 255.0f; // normalize and copy image
			*(h_Image + i + 1) = (unsigned char)*(imageData + index + 1) / 255.0f; // normalize and copy image
			*(h_Image + i + 2) = (unsigned char)*(imageData + index + 2) / 255.0f; // normalize and copy image
			*(h_Image + i + 3) = (unsigned char)0; // normalize and copy image
		}


		cudaMemcpyToArray( cu_array, 0, 0, h_Image, 4 * sizeof(float) * width * height, cudaMemcpyHostToDevice);
		checkCUDAError("ProcessImage: memcpy");


		// set texture parameters
		tex.addressMode[0] = cudaAddressModeWrap;
		tex.addressMode[1] = cudaAddressModeWrap;
		tex.filterMode = cudaFilterModeLinear;
		tex.normalized = false;    // access with normalized texture coordinates

		// Bind the array to the texture
		cudaBindTextureToArray( &tex, cu_array, &channelDesc);
		checkCUDAError("ProcessImage: Bind Texture");
				
		// Execute kernel.
		kernelLauncher( d_Image, width, height );
	
		// copy results back to h_C.
		cudaMemcpy( h_Image, d_Image, 3 * sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		checkCUDAError("ProcessImage: memcpy2");

		for(int i=0; i<3*width*height; i++)
		{
			// d_Image, 3 kanallý olduðu için doðrudan imageData'ya h_Image üzerinden kopyalanýr.
			*(imageData + i) = satchar(*(h_Image + i) * 255);
		}
	}

	virtual void ReleaseProcessing()
	{
		SingleCudaTexProcessor::ReleaseProcessing();

		cudaUnbindTexture( &tex );
		checkCUDAError("unbind tex");

		cudaFreeArray( cu_array );
		checkCUDAError("free device tex array");

		cudaFree( d_Image );
		checkCUDAError("free device image");
	
		cudaFreeHost( h_Image );
		checkCUDAError("free host image");
	}

};