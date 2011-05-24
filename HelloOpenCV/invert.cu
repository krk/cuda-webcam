#include "invert.h"

#define BLOCK_SIZE (32)

__global__
void gpuInvert(
	float* image,
	int width,
	int height
	)
{
	int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int k = blockIdx.y * BLOCK_SIZE + threadIdx.y;

	int cIdx = (i*width + k) * 3;

	*( image + cIdx ) = 1 - *( image + cIdx );
	*( image + cIdx + 1 ) = 1 - *( image + cIdx + 1 );
	*( image + cIdx + 2 ) = 1 - *( image + cIdx + 2 );
}

inline
void deviceInvertLaunch(
	int width,
	int height,
	float *d_Image,
	float* h_Image)
{
	 // launch kernel
	dim3 dimBlock( BLOCK_SIZE, BLOCK_SIZE );
    dim3 dimGrid( height / dimBlock.x, width / dimBlock.y );


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

    gpuInvert<<< dimGrid, dimBlock >>>( d_Image, width, height);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	
    // block until the device has completed
    cudaThreadSynchronize();
	
	printf("gpuInvert kernel time: %.3f ms\n", elapsedTime);

    // check if kernel execution generated an error
    // Check for any CUDA errors
    checkCUDAError("kernel invocation");
}

inline unsigned char satchar(float val)
{
	if(val<=0)
		return (unsigned char)0;
	if(val>=255)
		return (unsigned char)255;

	return (unsigned char)val;
}