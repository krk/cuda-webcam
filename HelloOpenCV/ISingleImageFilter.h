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