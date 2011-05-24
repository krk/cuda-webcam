#include "invert.h"

#define BLOCK_SIZE (32)

#define ENABLE_TIMING_CODE 0

__global__
void gpuBlur1(
	float* image,
	int width,
	int height
	)
{
	int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int k = blockIdx.y * BLOCK_SIZE + threadIdx.y;

	int cIdx = (i*width + k) * 3;
	int cIdxRight = (i*width + k + 1) * 3;
	int cIdxDown = ((i+1)*width + k) * 3;

	*( image + cIdx ) = (*( image + cIdx ) + *( image + cIdxRight ) + *( image + cIdxDown )) / 3;
	*( image + cIdx + 1 ) = 0;//*( image + cIdx + 1 );
	*( image + cIdx + 2 ) = 0;//*( image + cIdx + 2 );
}

void deviceBlur1Launch(
	float *d_Image,
	int width,
	int height
	)
{
	 // launch kernel
	dim3 dimBlock( BLOCK_SIZE, BLOCK_SIZE );
    dim3 dimGrid( height / dimBlock.x, width / dimBlock.y );

#if ENABLE_TIMING_CODE

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

#endif

    gpuBlur1<<< dimGrid, dimBlock >>>( d_Image, width, height);

#if ENABLE_TIMING_CODE
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

    // block until the device has completed
    cudaThreadSynchronize();
	
	printf("kernel time: %.3f ms\n", elapsedTime);
#endif

	cudaThreadSynchronize();

    // check if kernel execution generated an error
    // Check for any CUDA errors
    checkCUDAError("kernel invocation");
}
