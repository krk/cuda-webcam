#include "cudaCommon.h"

/**
	\file cudaCommon.cpp
	CUDA kullanan tüm sýnýflarda kullanýlan metodlarý içerilen dosya.
*/

/**
	CUDA metodlarýnýn dönüþ deðerlerinde hata kontrolü yapar.

	\param msg Hata halinde yazdýrýlacak mesaj.

	Dönüþ kodu cudaSuccess deðilse hata mesajý yazdýrýlýr ve program sonlandýrýlýr.
*/
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}