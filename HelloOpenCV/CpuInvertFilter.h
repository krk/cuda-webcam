#ifndef CPUINVERTFILTER_H_
#define CPUINVERTFILTER_H_

#include "common.h"

#include "SingleImageFilter.h"

/**
	\file CpuInvertFilter.h
	CpuInvertFilter sýnýfýnýn tanýmýný içerir.
*/

/**
	Cpu image invert filtre sýnýfý.

	Bu sýnýf SingleImageFilter sýnýfýný gerçekleyerek CPU üzerinde resmin negatifini almaya yarar.
*/

class CpuInvertFilter : public SingleImageFilter
{

public:

	CpuInvertFilter()
	{
	}

	/** Görüntünün RGB kanallarýnýn tersini alýr. */
	virtual void FilterImage(char* imageData)
	{
		// copy imageData to GPU.
		for(int i=0; i<3*width*height; i++)
		{
			*( imageData + i ) = ( unsigned char ) ( 255 - *( imageData + i ) ); // invert every pixel.
		}
	}

private:

	DISALLOW_COPY_AND_ASSIGN(CpuInvertFilter);

};

#endif // CPUINVERTFILTER_H_