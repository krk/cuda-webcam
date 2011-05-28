#pragma once

#include "SingleCudaTexFilter.h"

void SingleCudaTexFilter::InitFilter(int width, int height)
{
	SingleCudaFilter::InitFilter(width, height);

	/*
	allocate device texture memory
	*/

	// get texture reference.
	cudaGetTextureReference(&constTexRefPtr, textureSymbolName);
	checkCUDAError("get texture reference");
	
	texRefPtr = const_cast<textureReference*>( constTexRefPtr );

	channelDesc = cudaCreateChannelDesc<float4>();
	cudaMallocArray( &cu_array, &texRefPtr->channelDesc, width, height ); 
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


	// bind texture

	// set texture parameters
	texRefPtr->addressMode[0] = cudaAddressModeWrap;
	texRefPtr->addressMode[1] = cudaAddressModeWrap;

	texRefPtr->filterMode = cudaFilterModeLinear;
	texRefPtr->normalized = true;    // access with normalized texture coordinates



	checkCUDAError("FilterImage: Bind Texture");
}

void SingleCudaTexFilter::FilterImage(char* imageData)
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

	
	
	cudaMemcpyToArray( cu_array, 0, 0, h_Image, sizeof(float4) * width * height, cudaMemcpyHostToDevice);
	checkCUDAError("FilterImage: memcpy");

		// Bind the array to the texture
	cudaBindTextureToArray( texRefPtr, cu_array, &texRefPtr->channelDesc );



	// Execute kernel.
	kernelLauncher( d_Image, width, height );
	
	// copy results back to h_C.
	cudaMemcpy( h_Image, d_Image, 3 * sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	checkCUDAError("FilterImage: memcpy2");

	for(int i=0; i<3*width*height; i++)
	{
		// d_Image, 3 kanallý olduðu için doðrudan imageData'ya h_Image üzerinden kopyalanýr.
		*(imageData + i) = satchar(*(h_Image + i) * 255);
	}
}

void SingleCudaTexFilter::ReleaseFilter()
{
	SingleCudaFilter::ReleaseFilter();

	cudaUnbindTexture( texRefPtr );
	checkCUDAError("unbind tex");

	cudaFreeArray( cu_array );
	checkCUDAError("free device tex array");
}