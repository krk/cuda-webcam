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

#ifndef THRESHOLDFILTER_H_
#define THRESHOLDFILTER_H_

#include "common.h"

#include "SingleImageFilter.h"

/**
	\file ThresholdFilter.h
	ThresholdFilter sýnýfýnýn tanýmýný içerir.
*/

/**
	Eþik filtre sýnýfý.

	Bu sýnýf SingleImageFilter sýnýfýný gerçekleyerek CPU üzerinde resmin eþik deðerine göre siyah-beyaz görüntüsünü alýr.
*/

class ThresholdFilter : public SingleImageFilter
{

private:

	DISALLOW_COPY_AND_ASSIGN(ThresholdFilter);

	unsigned char threshold;
	
public:

	ThresholdFilter(unsigned char threshold)
	{
		this->threshold = threshold;
	}
	
	/** Görüntünün eþik deðerine göre siyah-beyazýný alýr. */
	virtual void FilterImage(char* imageData)
	{
		for(int i=0; i<width*height; i++)
		{
			// Y=0.3RED+0.59GREEN+0.11Blue

			unsigned char b = *( imageData + i * 3 + 0 );
			unsigned char g = *( imageData + i * 3 + 1 );
			unsigned char r = *( imageData + i * 3 + 2 );

			unsigned char grayscale = ( unsigned char ) ( 0.3f * r + 0.59f * g + 0.11f * b );

			unsigned char result = grayscale > threshold ? 255 : 0;

			*( imageData + i * 3 + 0 ) = result;
			*( imageData + i * 3 + 1 ) = result;
			*( imageData + i * 3 + 2 ) = result;
		}
	}
};

#endif // THRESHOLDFILTER_H_