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

#ifndef CPUCCL_H_
#define CPUCCL_H_

#include "common.h"

#include "SingleImageFilter.h"

#include "boost\pending\disjoint_sets.hpp"
#include "boost\unordered_set.hpp"

/**
	\file CpuCCLFilter.h
	CpuCCLFilter sýnýfýnýn tanýmýný içerir.
*/

/**
	Cpu Connected Component Labeling filtre sýnýfý.

	Bu sýnýf SingleImageFilter sýnýfýný gerçekleyerek CPU üzerinde resmin birleþik nesnelerini ayýrdeder.
*/
class CpuCCLFilter : public SingleImageFilter
{

private:

	DISALLOW_COPY_AND_ASSIGN(CpuCCLFilter);
	
	char* auxImageData; // grayscale holder.
	char* auxImageDataOut; // output.
	int* auxOutLabelBitmap;

	template<bool eightConnected>
	void findConnectedComponents(int width, int height, int rowStride, char* imageData, char* outImageData, int* outLabelBitmap);

public:

	CpuCCLFilter()
	{
	}

	virtual void InitFilter(int width, int height, int rowStride)
	{
		SingleImageFilter::InitFilter(width, height, rowStride);	

		auxImageData = (char*) malloc( sizeof(char) * rowStride / 3 * height );
		auxImageDataOut = (char*) malloc( sizeof(char) * rowStride * height );
		auxOutLabelBitmap = (int*) malloc( sizeof(int) * rowStride * height);
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

				*( auxImageData + idx ) = ( unsigned char ) ( 0.3f * r + 0.59f * g + 0.11f * b ); // grayscale
			}
		}
		
		findConnectedComponents<true>( width, height, rowStride / 3, auxImageData, auxImageDataOut, auxOutLabelBitmap );

		memcpy( imageData, auxImageDataOut, sizeof(char) * rowStride * height );

		/*for(int i=0; i<height; i++)
		{
			for(int j=0; j<width; j++)
			{
				int idx = i * (rowStride / 3) + j;

				*( imageData + idx * 3 + 0 ) = *( auxImageDataOut + idx * 3 + 0 );
				*( imageData + idx * 3 + 1 ) = *( auxImageDataOut + idx * 3 + 1 );
				*( imageData + idx * 3 + 2 ) = *( auxImageDataOut + idx * 3 + 2 );
			}
		}*/
	}

	virtual void ReleaseFilter()
	{
		if( this->isReleased )
			return;

		SingleImageFilter::ReleaseFilter();

		if( auxImageData )
		{
			free( auxImageData );
		}

		if( auxImageDataOut )
		{
			free( auxImageDataOut );
		}

		if( auxOutLabelBitmap )
		{
			free( auxOutLabelBitmap );
		}
	}

};

/** Renkler birbirine 10 birim yakýnsa true döndürür. */
inline bool isSameColor(char c1, char c2)
{
	if(c1==0)
		return false;

	if(c2==0)
		return false;

	int uc1 = (unsigned char)c1;
	int uc2 = (unsigned char)c2;

	return abs(uc1-uc2) < 10;	
}


inline float hue2rgb(float p, float q, float t)
{
	if(t < 0) t += 1;
	if(t > 1) t -= 1;
	if(t < 1.0f/6) return p + (q - p) * 6 * t;
	if(t < 1.0f/2) return q;
	if(t < 2.0f/3) return p + (q - p) * (2.0f/3 - t) * 6;
	return p;
}

/** HSL renklerini RGB uzayýna çevirir. */
void hsl_to_rgb(float h, float s, float l, int* R, int* G, int* B)
{
	if ( s == 0 )                       //HSL from 0 to 1
	{
	   *R = l * 255;                      //RGB results from 0 to 255
	   *G = l * 255;
	   *B = l * 255;
	}
	else
	{
		float q = l < 0.5f ? l * (1 + s) : l + s - l * s;
		float p = 2 * l - q;

		*R = 255 * hue2rgb(p, q, h + 1.0f/3);
		*G = 255 * hue2rgb(p, q, h);
		*B = 255 * hue2rgb(p, q, h - 1.0f/3);
	}
}

/**
	\ref ptKernelLauncher tipinde metod.

	\param width Görüntünün piksel olarak geniþliði
	\param height Görüntünün piksel olarak yüksekliði
	\param rowStride Görüntünün hafýzadaki byte olarak geniþliði.
	\param imageData Görüntü verisi.
	\param outImageData Etiket görüntü verisi.
	\param outLabelBitmap Etiket integer bitmap.

	Görüntüdeki birleþmiþ birimleri bulur. Connected Component Labeler.
*/
template<bool eightConnected>
void CpuCCLFilter::findConnectedComponents(int width, int height, int rowStride, char* imageData, char* outImageData, int* outLabelBitmap)
{
	// label bitmap	

	// disjoint set için tip ve map tanýmlarý.
	typedef std::map<short, std::size_t> rank_t; // element order.
	typedef std::map<short, short> parent_t;

	rank_t rank_map;
	parent_t parent_map;

	boost::associative_property_map<rank_t>   rank_pmap( rank_map );
	boost::associative_property_map<parent_t> parent_pmap( parent_map );

	// disjoint sets yaratýlýr.
	boost::disjoint_sets<boost::associative_property_map<rank_t>, boost::associative_property_map<parent_t>> 
		ds( rank_pmap, parent_pmap );

	boost::unordered_set<short> elements;

	short labelNumber = 0;

	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			outLabelBitmap[i*rowStride + j] = 0;
		}
	}


	// tüm pikseller gezilir
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			// if pixel is not background.
			if( imageData[i*rowStride + j] != 0 )
			{
				if(j > 0 && i > 0 &&
					isSameColor( imageData[i*rowStride + j - 1], imageData[(i - 1)*rowStride + j]) // same value
					&& isSameColor( imageData[i*rowStride + j - 1], imageData[i*rowStride + j] )
				)
				{
					// west ve north ayný
					ds.union_set( outLabelBitmap[i*rowStride + j - 1], outLabelBitmap[(i - 1)*rowStride + j] );
					outLabelBitmap[i*rowStride + j] = outLabelBitmap[i*rowStride + j - 1];
				}

				if(j==0 && i > 0 && isSameColor( imageData[i*rowStride + j], imageData[(i - 1)*rowStride + j] ))
				{
					outLabelBitmap[i*rowStride + j] = outLabelBitmap[(i-1)*rowStride + j];
				}

				// West pixel
				if(j > 0 && isSameColor( imageData[i*rowStride + j - 1], imageData[i*rowStride + j] ))
				{
					// west pixel ile ayný label.
					outLabelBitmap[i*rowStride + j] = outLabelBitmap[i*rowStride + j - 1];
				}
				
				// west farklý north ayný.
				if(
					(j > 0 && !isSameColor( imageData[i*rowStride + j - 1], imageData[i*rowStride + j] )) // west different value
					&& (i > 0 && isSameColor( imageData[(i-1)*rowStride + j], imageData[i*rowStride + j] )) // north same value					
				)
				{
					// north ile ayný
					outLabelBitmap[i*rowStride + j] = outLabelBitmap[(i-1)*rowStride + j];					
				}

				// west ve north farklý
				if(   ((j > 0 && !isSameColor( imageData[i*rowStride + j - 1], imageData[i*rowStride + j] )) || j == 0) // west different value
					&& ((i > 0 && !isSameColor( imageData[(i-1)*rowStride + j], imageData[i*rowStride + j] )) || i == 0) // north different value					
				)
				{
					labelNumber++;
					ds.make_set(labelNumber);
					elements.insert(labelNumber);
					outLabelBitmap[i*rowStride + j] = labelNumber;
				}

				// northeast ve northwest kontrol edilir. (çaprazlar).
				if(eightConnected)
				{
					// northwest
					if(j > 0 && i > 0
						&& isSameColor( imageData[(i-1)*rowStride + j - 1], imageData[i*rowStride + j] )
					)
					{
						// northwest ile ayný
						ds.union_set( outLabelBitmap[(i-1)*rowStride + j - 1], outLabelBitmap[i*rowStride + j] );
					}

					// northeast
					if(j+1 < width && i > 0
						&& isSameColor( imageData[(i-1)*rowStride + j + 1], imageData[i*rowStride + j] )
					)
					{
						// northeast ile ayný
						ds.union_set( outLabelBitmap[(i-1)*rowStride + j + 1], outLabelBitmap[i*rowStride + j] );
					}
				}
			}
		}
	}

	int cnt = ds.count_sets(elements.begin(), elements.end());
	
	printf("Component count: %i\n", cnt);
	
	// second pass - label output image and colorize.
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			int labelNo = ds.find_set( outLabelBitmap[i*rowStride + j] ); // pikselin etiketi bulunur.
			int R =0, G=0, B=0;
			
			// etiketler renklendirilir.
			hsl_to_rgb( 1.0f * labelNo / labelNumber, .8f + .2f * labelNo / labelNumber, .75f, &R, &G, &B );
					
			int idx = i * rowStride + j;

			outImageData[idx * 3 + 0] = (char)(B);
			outImageData[idx * 3 + 1] = (char)(G);
			outImageData[idx * 3 + 2] = (char)(R);
		}
	}	
}

#endif // CPUCCL_H_