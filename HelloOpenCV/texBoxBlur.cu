#ifndef TEX_INVERT_CU
#define TEX_INVERT_CU

#include "texBoxBlur.h"

texture<float4, 2, cudaReadModeElementType> texBlur1;

#define BLOCK_SIZE_X (32)
#define BLOCK_SIZE_Y (32)

#define ENABLE_TIMING_CODE 0

__global__
void gpuTexBoxBlur(
	float* image,
	int width,
	int height
	)
{
	int row = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
	int col = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;

	int cIdx = ( row * width + col ) * 3; // 3 ile çarpým RGB için, linearIndex.

	float tw = 1.0f / width; // Texture kordinatlarýnda 1 pixel geniþlik.
	float th = 1.0f / height; // Texture kordinatlarýnda 1 pixel yükseklik.

	float tu = ( float )col * tw;
	float tv = ( float )row * th;


	float4 texVal = make_float4(0, 0, 0, 0);

	tw *= 8;
	th *= 8;

#pragma unroll
	for(int i = 0; i < 5; i++)
	{
#pragma unroll
		for(int j = 0; j < 5; j++)
		{			
			texVal += tex2D( texBlur1, tu + ( i - 2 ) * tw, tv + ( j - 2 ) * th );
		}
	}

	texVal *= 1.0f / 25;


	*( image + cIdx )     = texVal.x;
	*( image + cIdx + 1 ) = texVal.y;
	*( image + cIdx + 2 ) = texVal.z;
}

void deviceTexBoxBlurLaunch(
	float *d_Image,
	int width,
	int height
	)
{
	 // launch kernel
	dim3 dimBlock( BLOCK_SIZE_X, BLOCK_SIZE_Y );
    dim3 dimGrid( width / dimBlock.x, height / dimBlock.y );

#if ENABLE_TIMING_CODE

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

#endif

    gpuTexBoxBlur<<< dimGrid, dimBlock >>>( d_Image, width, height);

#if ENABLE_TIMING_CODE
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

    // block until the device has completed
    cudaThreadSynchronize();
	
	printf("gpuTexBoxBlur kernel time: %.3f ms\n", elapsedTime);
#endif

	cudaThreadSynchronize();

    // check if kernel execution generated an error
    // Check for any CUDA errors
    checkCUDAError("kernel invocation");
}


#endif