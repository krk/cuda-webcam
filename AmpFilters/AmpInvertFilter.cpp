#include "AmpInvertFilter.h"
#include "amp.h"

using namespace concurrency;

/**
	Cpu image invert filtre sýnýfý.

	Bu sýnýf SingleImageFilter sýnýfýný gerçekleyerek CPU üzerinde resmin negatifini almaya yarar.
*/

/** Görüntünün RGB kanallarýnýn tersini alýr. */
void AmpInvertFilter::FilterImage(char* imageData)
{
	auto data = reinterpret_cast<unsigned int*>(imageData);

	const int size = 3*width*height;
	array_view<const unsigned int, 1> img(size, data);
	array_view<unsigned int, 1> result(size, data);
	result.discard_data();

	/*
	for(int i=0; i<3*width*height; i++)
	{
		*( imageData + i ) = ( unsigned char ) ( 255 - *( imageData + i ) ); // her pikselin her kanalýnýn negatifini al.
	}*/

	parallel_for_each( 
        // Define the compute domain, which is the set of threads that are created.
        img.extent, 
        // Define the code to run on each thread on the accelerator.
        [=](index<1> idx) restrict(amp)
		{
			result[idx] = 255 - img[idx];
		}
    );
}
