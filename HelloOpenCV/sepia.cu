#include "sepia.h"

/**
	\file sepia.cu
	CUDA sepia kernelinin launcher metodunu ve kernelini tanýmlar.
*/

/** Kernel 1 griddeki blok boyutu ( BLOCK_SIZE x BLOCK_SIZE kare bloklar ). */
#define BLOCK_SIZE (32)

/** GPU zamanýný ölçmek için 1 yapýnýz. */
#define ENABLE_TIMING_CODE 0

/**	
	Görüntünün sepia tonlamasýný hesaplayan kernel.

	\param image [0, 1] aralýðýna normalize edilmiþ, BGR kanal sýralý görüntünün GPU belleðindeki adresi.
	\param width Görüntünün piksel olarak geniþliði
	\param height Görüntünün piksel olarak yüksekliði

	
	Metod GPU üzerinde çalýþýr, çýktýsýný image parametresinin üzerine yazar.
	*/
__global__
void gpuSepia(
	float* image,
	int width,
	int height
	)
{
	int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	int cIdx = ( row * width + col ) * 3; // 3 ile çarpým RGB için, linearIndex.

	// normalize edilmiþ pikselleri 1'den çýkarttýðýmýzda görüntünün negatifini almýþ oluruz.

	float b = *( image + cIdx     );
	float g = *( image + cIdx + 1 );
	float r = *( image + cIdx + 2 );


		/*outputRed = (inputRed * .393) + (inputGreen *.769) + (inputBlue * .189)

outputGreen = (inputRed * .349) + (inputGreen *.686) + (inputBlue * .168)

outputBlue = (inputRed * .272) + (inputGreen *.534) + (inputBlue * .131)*/

	*( image + cIdx     ) = .272f * r + .534f * g + .131f * b; // Red kanalý
	*( image + cIdx + 1 ) = .349f * r + .686f * g + .168f * b; // Green kanalý
	*( image + cIdx + 2 ) = .393f * r + .769f * g + .189f * b; // Blue kanalý
}

/**
	\ref ptKernelLauncher tipinde metod.

	\param d_Image [0, 1] aralýðýna normalize edilmiþ, BGR kanal sýralý görüntünün GPU belleðindeki adresi.
	\param width Görüntünün piksel olarak geniþliði
	\param height Görüntünün piksel olarak yüksekliði

	\ref gpuSepia kernelini Grid ve Block boyutlarýný ayarlayarak çaðýran metod.
*/
void deviceSepiaLaunch(
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

    gpuSepia<<< dimGrid, dimBlock >>>( d_Image, width, height);

#if ENABLE_TIMING_CODE
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

    // block until the device has completed
    cudaThreadSynchronize();
	
	printf("gpuSepia kernel time: %.3f ms\n", elapsedTime);
#endif

	cudaThreadSynchronize();

    // check if kernel execution generated an error
    // Check for any CUDA errors
    checkCUDAError("kernel invocation");
}
