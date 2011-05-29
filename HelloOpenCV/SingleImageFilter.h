#ifndef SINGLEIMAGEFILTER_H_
#define SINGLEIMAGEFILTER_H_

/** 
	\file SingleImageFilter.h
	SingleImageFilter sýnýfýnýn tanýmýný içeren baþlýk dosyasý.
*/

#include "common.h"

#include "ISingleImageFilter.h"

/**
	ISingleImageFilter arayüzünü FilterImage metodu hariç gerçekleyen abstract sýnýf.
*/
class SingleImageFilter : public ISingleImageFilter
{
private:
	DISALLOW_COPY_AND_ASSIGN(SingleImageFilter);
	
protected:
	bool isInited;
	bool isReleased;
	int height;
	int width;

public:

	virtual ~SingleImageFilter()
	{
	}

	SingleImageFilter(void)
		: isInited(false), 
		isReleased(false), 
		height(0), 
		width(0)
	{
		
	}

	virtual void InitFilter(int width, int height)
	{
		this->width = width;
		this->height = height;
		this->isInited = true;
	}

	virtual void ReleaseFilter()
	{
		this->isReleased = true;
	}
};

#endif SINGLEIMAGEFILTER_H_