// Copyright (c) 2011 Kerem KAT 
// 
// http://dissipatedheat.com/
// Do not hesisate to contact me about usage of the code or to make comments 
// about the code. Your feedback will be appreciated.
// keremkat<@>gmail<.>com
//
// Kodun kullanýmý hakkýnda veya yorum yapmak için benimle iletiþim kurmaktan
// çekinmeyiniz. Geri bildirimleriniz deðerlendirilecektir.
// keremkat<@>gmail<.>com
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to 
// deal in the Software without restriction, including without limitation the 
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
// sell copies of the Software, and to permit persons to whom the Software is 
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#ifndef CPUMOVINGAVERAGEFILTER_H_
#define CPUMOVINGAVERAGEFILTER_H_

#include "common.h"

#include "SingleImageFilter.h"

/**
	\file CpuMovingAverageFilter.h
	CpuMovingAverageFilter sýnýfýnýn tanýmýný içerir.
*/

/**
	Cpu Moving Average filtre sýnýfý.

	Bu sýnýf SingleImageFilter sýnýfýný gerçekleyerek CPU üzerinde framelerin ortalamasýný alýr.
*/

class CpuMovingAverageFilter : public SingleImageFilter
{

private:

	DISALLOW_COPY_AND_ASSIGN(CpuMovingAverageFilter);
	
	float* auxMovingAverage;
	int auxFramesInAverage;

	int framesToAverage;

public:

	CpuMovingAverageFilter(int framesToAverage)
	{
		this->framesToAverage = framesToAverage;
	}

	virtual void InitFilter(int width, int height, int rowStride)
	{
		SingleImageFilter::InitFilter(width, height, rowStride);	

		auxMovingAverage = (float*) malloc( sizeof(float) * rowStride / 3 * height );		
		auxFramesInAverage = 0;

		memset( auxMovingAverage, 0, sizeof(float) * rowStride / 3 * height );
	}

	/** Birleþik elemanlarý bularak(max 255 adet) her birine ayrý bir gri tonu atar. */
	virtual void FilterImage(char* imageData)
	{
		for(int i=0; i<height; i++)
		{
			for(int j=0; j<width; j++)
			{
				int idx = i * (rowStride / 3) + j;

				unsigned char b = *( imageData + idx * 3 + 0 );
				unsigned char g = *( imageData + idx * 3 + 1 );
				unsigned char r = *( imageData + idx * 3 + 2 );

				float mvAvgR = *( auxMovingAverage + idx + 0 );
				float mvAvgG = *( auxMovingAverage + idx + 1 );
				float mvAvgB = *( auxMovingAverage + idx + 2 );

				float resultB = auxFramesInAverage > 0 ? ( 1.1f * b + .9f * auxFramesInAverage * mvAvgB ) / ( auxFramesInAverage + 1 ) : b;
				float resultG = auxFramesInAverage > 0 ? ( 1.1f * g + .9f * auxFramesInAverage * mvAvgG ) / ( auxFramesInAverage + 1 ) : g;
				float resultR = auxFramesInAverage > 0 ? ( 1.1f * r + .9f * auxFramesInAverage * mvAvgR ) / ( auxFramesInAverage + 1 ) : r;

				*( auxMovingAverage + idx + 0 ) = resultB;
				*( auxMovingAverage + idx + 1 ) = resultG;
				*( auxMovingAverage + idx + 2 ) = resultR;

				if( auxFramesInAverage < framesToAverage )
					continue;

				*( imageData + idx * 3 + 0 ) = ( unsigned char ) ( resultB );
				*( imageData + idx * 3 + 1 ) = ( unsigned char ) ( resultG );
				*( imageData + idx * 3 + 2 ) = ( unsigned char ) ( resultR );
			}
		}
		
		if( auxFramesInAverage < framesToAverage )
		{
			auxFramesInAverage++;
		}
	}

	virtual void ReleaseFilter()
	{
		if( this->isReleased )
			return;

		SingleImageFilter::ReleaseFilter();

		if( auxMovingAverage )
		{
			free( auxMovingAverage );
		}

		auxFramesInAverage = 0;
	}

};

#endif // CPUMOVINGAVERAGEFILTER_H_