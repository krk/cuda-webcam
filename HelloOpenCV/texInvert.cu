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

#ifndef TEX_INVERT_CU
#define TEX_INVERT_CU

/**
	\file texInvert.cu
	CUDA texture invert kernelinin launcher metodunu ve kernelini tanýmlar.
*/

#include "texInvert.h"

texture<float4, 2, cudaReadModeElementType> texInvert1; /**< Kernelde kullanýlan texture sembolü. */

#define BLOCK_SIZE (32) /**< Blok boyutu ( BLOCK_SIZE x BLOCK_SIZE kare blok ). */

/** GPU zamanýný ölçmek için 1 yapýnýz. */
#define ENABLE_TIMING_CODE 0

/**	
	Texture kullanarak görüntünün negatifini alan kernel.

	\param image [0, 1] aralýðýna normalize edilmiþ, BGR kanal sýralý görüntünün GPU belleðindeki adresi.
	\param width Görüntünün piksel olarak geniþliði
	\param height Görüntünün piksel olarak yüksekliði

	
	Metod GPU üzerinde çalýþýr, çýktýsýný image parametresinin üzerine yazar.
*/
__global__
void gpuTexInvert(
	float* image,
	int width,
	int height
	)
{
	int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	int cIdx = ( row * width + col ) * 3; // 3 ile çarpým RGB için, linearIndex.


	// threade ait kordinatýn texture uzayýndaki kordinatý bulunur.
	float tu = (float)col / width;
	float tv = (float)row / height;

	// Texture üzerinden görüntü verisi okunur.
	float4 texVal = tex2D( texInvert1, tu, tv );

	// Texture deðerleri 1'den çýkartýlarak global belleðe yazýlýr.
	*( image + cIdx )     = 1 - texVal.x;
	*( image + cIdx + 1 ) = 1 - texVal.y;
	*( image + cIdx + 2 ) = 1 - texVal.z;
}

/**
	\ref ptKernelLauncher tipinde metod.

	\param d_Image [0, 1] aralýðýna normalize edilmiþ, BGR kanal sýralý görüntünün GPU belleðindeki adresi.
	\param width Görüntünün piksel olarak geniþliði
	\param height Görüntünün piksel olarak yüksekliði

	\ref gpuTexInvert kernelini Grid ve Block boyutlarýný ayarlayarak çaðýran metod.
*/
void deviceTexInvertLaunch(
	float *d_Image,
	int width,
	int height
	)
{
	 // launch kernel
	dim3 dimBlock( BLOCK_SIZE, BLOCK_SIZE );
    dim3 dimGrid( width / dimBlock.x, height / dimBlock.y );

#if ENABLE_TIMING_CODE

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

#endif

    gpuTexInvert<<< dimGrid, dimBlock >>>( d_Image, width, height);

#if ENABLE_TIMING_CODE
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

    // block until the device has completed
    cudaThreadSynchronize();
	
	printf("gpuInvert kernel time: %.3f ms\n", elapsedTime);
#endif

	cudaThreadSynchronize();

    // check if kernel execution generated an error
    // Check for any CUDA errors
    checkCUDAError("kernel invocation");
}


#endif