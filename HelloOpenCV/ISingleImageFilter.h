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

#ifndef ISINGLEIMAGEFILTER_H_
#define ISINGLEIMAGEFILTER_H_

/**
	\file ISingleImageFilter.h
	ISingleImageFilter arayüzünün tanýmýný içerir.
*/

/**
	Resim filtreleme iþlemleri için ISingleImageFilter arayüzü.

	Arayüzdeki FilterImage metodunu gerçekleyen sýnýflar görüntü iþleme yapabilirler. 
	Filtreleme iþlemine baþlamadan önce bir kere InitFilter metodu çaðrýlýr, 
	filtreleme tekrar yapýlmayacaksa ReleaseFilter metodu çaðrýlýr.
*/

class ISingleImageFilter
{

public:

	/**
		Görüntü ile ilgili alloc ve benzeri iþlemlerin yapýlmasýný saðlar. 	

		\param width Görüntünün piksel geniþliði.
		\param height Görüntünün piksel yüksekliði.
	*/
	virtual void InitFilter(int width, int height) = 0;

	/** 
		Görüntünün yerinde(in-place) iþlenmesi için çaðrýlan metod. 

		\param imageData Görüntünün BGR kanal sýralý bellekteki adresi.
	*/
	virtual void FilterImage(char* imageData) = 0;

	/** Görüntü ile ilgili kaynaklarýn býrakýlmasýný saðlar. */
	virtual void ReleaseFilter() = 0;

};

#endif // ISINGLEIMAGEFILTER_H_