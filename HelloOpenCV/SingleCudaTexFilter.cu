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

#include "SingleCudaTexFilter.h"

/** 
	\file SingleCudaTexFilter.cu
	SingleCudaTexFilter sýnýfýnýn tanýmýný içeren dosya.
*/

void SingleCudaTexFilter::InitFilter(int width, int height)
{
	SingleCudaFilter::InitFilter(width, height);

	/*
	allocate device texture memory
	*/

	// cudaGetTextureReference, istediði gibi const texture referansý ile çaðrýlýr.
	cudaGetTextureReference(&constTexRefPtr, textureSymbolName);
	checkCUDAError("get texture reference");
	
	// const olan referans, const olmayan referansa dönüþtürülerek saklanýr.
	texRefPtr = const_cast<textureReference*>( constTexRefPtr );

	channelDesc = cudaCreateChannelDesc<float4>();

	// const olmayan texture referansýnýn kullanýmý.
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

	
	// cu_array, texturea atanýr.
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