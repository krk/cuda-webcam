#ifndef TEX_INVERT_CU
#define TEX_INVERT_CU

/**
	\file texAbsDiff.cu
	CUDA texture absolute difference kernelinin launcher metodunu ve kernelini tanýmlar.
*/

#include "texAbsDiff.h"

texture<float4, 2, cudaReadModeElementType> texAbsDiff1; /**< Kernelde kullanýlan texture sembolü. */

#define BLOCK_SIZE_X (32) /**< Yatay blok boyutu */
#define BLOCK_SIZE_Y (32) /**< Düþey blok boyutu */

/** GPU zamanýný ölçmek için 1 yapýnýz. */
#define ENABLE_TIMING_CODE 1


/**
	Mutlak deðer metodu.

	\param a Mutlak deðeri alýnacak float4 deðeri.

	a parametresinin her bir elemanýnýn mutlak deðerini alarak yeni bir float4 döndürür.
*/
inline __host__ __device__ float4 abs( float4 a )
{	
	return make_float4( fabsf( a.x ), fabsf( a.y ), fabsf( a.z ), fabsf( a.w ) );
}

/**	
	Texture kullanarak görüntünün komþu pikseller ile mutlak farkýný alan kernel.

	\param image [0, 1] aralýðýna normalize edilmiþ, BGR kanal sýralý görüntünün GPU belleðindeki adresi.
	\param width Görüntünün piksel olarak geniþliði
	\param height Görüntünün piksel olarak yüksekliði

	
	Metod GPU üzerinde çalýþýr, çýktýsýný image parametresinin üzerine yazar.

	*/
__global__
void gpuTexAbsDiff(
	float* image,
	int width,
	int height
	)
{
	// threade ait satýr ve sütunu hesapla.
	int row = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
	int col = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;

	// global memorydeki imaj için indisi hesapla.
	int cIdx = ( row * width + col ) * 3; // 3 ile çarpým RGB için, linearIndex.

	float tw = 1.0f / width; // Texture kordinatlarýnda 1 pixel geniþlik.
	float th = 1.0f / height; // Texture kordinatlarýnda 1 pixel yükseklik.

	// merkez piksel kordinatýn normalize texel kordinatlarýný hesapla.
	float tu = ( float )col * tw;
	float tv = ( float )row * th;

	/*tw *= 4;
	th *= 4;*/
	
	// row, col etrafýndaki 4 ana yöndeki texelden farklarýný al.
	float4 texVal1 = tex2D( texAbsDiff1, tu, tv ) - tex2D( texAbsDiff1, tu + tw, tv + th );
	float4 texVal2 = tex2D( texAbsDiff1, tu, tv ) - tex2D( texAbsDiff1, tu - tw, tv - th );
	float4 texVal3 = tex2D( texAbsDiff1, tu, tv ) - tex2D( texAbsDiff1, tu + tw, tv - th );
	float4 texVal4 = tex2D( texAbsDiff1, tu, tv ) - tex2D( texAbsDiff1, tu - tw, tv + th );

	// 4 ana yöndeki farklarýn mutlak deðerlerinin ortalamasýný al.
	float4 texVal = 0.25f * ( abs( texVal1 ) + abs( texVal2 ) + abs( texVal3 ) + abs( texVal4 ) );

	// global memorydeki imaja hesaplanan deðerleri aktar.
	*( image + cIdx )     = texVal.x;
	*( image + cIdx + 1 ) = texVal.y;
	*( image + cIdx + 2 ) = texVal.z;
}

/**
	\ref ptKernelLauncher tipinde metod.

	\param d_Image [0, 1] aralýðýna normalize edilmiþ, BGR kanal sýralý görüntünün GPU belleðindeki adresi.
	\param width Görüntünün piksel olarak geniþliði
	\param height Görüntünün piksel olarak yüksekliði

	\ref gpuTexAbsDiff kernelini Grid ve Block boyutlarýný ayarlayarak çaðýran metod.
*/
void deviceTexAbsDiffLaunch(
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

    gpuTexAbsDiff<<< dimGrid, dimBlock >>>( d_Image, width, height);

#if ENABLE_TIMING_CODE
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

    // block until the device has completed
    cudaThreadSynchronize();
	
	printf("gpuTexAbsDiff kernel time: %.3f ms\n", elapsedTime);
#endif

	cudaThreadSynchronize();

    // check if kernel execution generated an error
    // Check for any CUDA errors
    checkCUDAError("kernel invocation");
}


#endif