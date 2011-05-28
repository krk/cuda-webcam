#include "cudaCommon.h"
#include "SingleImageFilter.h"

#include <vector>

#pragma once

class SingleImageFilterChain : public SingleImageFilter
{
private:
	std::vector<ISingleImageFilter*> vecFilters;

public:

	SingleImageFilterChain()		
	{		
	}

	void AppendFilter(ISingleImageFilter* filter)
	{
		assert( filter && "AppendFilter: filter is invalid." );

		vecFilters.push_back( filter );

		if ( isInited )
		{
			filter->InitFilter( width, height );
		}
	}

	void RemoveLastFilter()
	{
		ISingleImageFilter* lastFilter = vecFilters.back();
		assert( lastFilter && "RemoveLastFilter: lastFilter is invalid." );

		lastFilter->ReleaseFilter();

		vecFilters.pop_back();
	}

	void RemoveAll()
	{
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