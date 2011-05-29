#ifndef SINGLEIMAGEFILTERCHAIN_H_
#define SINGLEIMAGEFILTERCHAIN_H_

/** 
	\file SingleImageFilterChain.h
	SingleImageFilterChain sýnýfýnýn tanýmýný içeren baþlýk dosyasý.
*/

#include <vector>

#include "common.h"

#include "cudaCommon.h"

#include "SingleImageFilter.h"

/**
	Bir veya birden fazla filtrenin tek bir filtre gibi uygulanmasýný saðlayan sýnýf.

	FilterImage metodu çaðrýldýðýnda AppendFilter metodu ile eklenen filtreleri eklenme sýrasýna uygun olarak zincir gibi CPU veya GPU üzerinde çalýþtýrýr.
	Eklenebilen filtrelerin tipi ISingleImageFilter tipindedir.
*/
class SingleImageFilterChain : public SingleImageFilter
{
private:
	std::vector<ISingleImageFilter*> vecFilters;

	DISALLOW_COPY_AND_ASSIGN(SingleImageFilterChain);

public:

	/**
		SingleImageFilterChain yaratýcýsý.	
	*/
	SingleImageFilterChain()		
	{		
	}

	/**
		Zincire filtre ekler.

		\param filter Eklenecek filtre.
	*/
	void AppendFilter(ISingleImageFilter* filter)
	{
		assert( filter && "AppendFilter: filter is invalid." );

		vecFilters.push_back( filter );

		if ( isInited )
		{
			filter->InitFilter( width, height );
		}
	}

	/**
		Son eklenen filtreyi zincirden çýkartýr.

		Çýkartýlan filtrenin ReleaseFilter metodu çaðrýlýr.
	*/
	void RemoveLastFilter()
	{
		ISingleImageFilter* lastFilter = vecFilters.back();
		assert( lastFilter && "RemoveLastFilter: lastFilter is invalid." );

		lastFilter->ReleaseFilter();

		vecFilters.pop_back();
	}

	/**
		Tüm filtreleri siler.

		Çýkartýlan filtrenin ReleaseFilter metodu çaðrýlýr.
	*/
	void RemoveAll()
	{
		vector<ISingleImageFilter*>::const_iterator fi;
		vector<ISingleImageFilter*>::const_iterator fEnd;

		for (fi = vecFilters.begin(), fEnd = vecFilters.end(); fi != fEnd; fi++)
		{			
			(*fi)->ReleaseFilter();
		}

		vecFilters.clear();
	}
	
	virtual void InitFilter(int width, int height)
	{
		SingleImageFilter::InitFilter(width, height);	

		vector<ISingleImageFilter*>::const_iterator fi;
		vector<ISingleImageFilter*>::const_iterator fEnd;

		for (fi = vecFilters.begin(), fEnd = vecFilters.end(); fi != fEnd; fi++)
		{			
			(*fi)->InitFilter( width, height );
		}
	}

	virtual void FilterImage(char* imageData)
	{
		vector<ISingleImageFilter*>::const_iterator fi;
		vector<ISingleImageFilter*>::const_iterator fEnd;

		for (fi = vecFilters.begin(), fEnd = vecFilters.end(); fi != fEnd; fi++)
		{
			(*fi)->FilterImage( imageData );
		}
	}

	virtual void ReleaseFilter()
	{
		SingleImageFilter::ReleaseFilter();

		vector<ISingleImageFilter*>::const_iterator fi;
		vector<ISingleImageFilter*>::const_iterator fEnd;

		for (fi = vecFilters.begin(), fEnd = vecFilters.end(); fi != fEnd; fi++)
		{
			(*fi)->ReleaseFilter();
		}
	}

};

#endif SINGLEIMAGEFILTERCHAIN_H_