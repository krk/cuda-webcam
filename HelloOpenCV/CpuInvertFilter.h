#include "SingleImageProcessor.h"

#pragma once

class CpuInvertFilter : public SingleImageProcessor
{
public:

	virtual void ProcessImage(char* imageData)
	{
		// copy imageData to GPU.
		for(int i=0; i<3*width*height; i++)
		{
			*( imageData + i ) = ( unsigned char ) ( 255 - *( imageData + i ) ); // invert every pixel.
		}
	}

};

