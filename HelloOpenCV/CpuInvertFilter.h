#include "SingleImageFilter.h"

#pragma once

class CpuInvertFilter : public SingleImageFilter
{
public:

	virtual void FilterImage(char* imageData)
	{
		// copy imageData to GPU.
		for(int i=0; i<3*width*height; i++)
		{
			*( imageData + i ) = ( unsigned char ) ( 255 - *( imageData + i ) ); // invert every pixel.
		}
	}

};

