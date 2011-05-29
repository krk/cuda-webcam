#ifndef IDENTITYFILTER_H_
#define IDENTITYFILTER_H_

#include "common.h"
#include "SingleImageFilter.h"

/**
	\file IdentityFilter.h
	IdentityFilter sýnýfýnýn tanýmýný içerir.
*/

/**
	Bu sýnýf SingleImageFilter sýnýfýný boþ gerçekleyerek görüntüde hiç bir deðiþiklik yapmaz.
*/

class IdentityFilter : public SingleImageFilter
{
public:
	IdentityFilter()
	{
	}

	/** Görüntüde deðiþiklik yapmadan çýkar. */
	virtual void FilterImage(char* imageData)
	{
		return; // imajý deðiþtirmeden dön.
	}

private:
	DISALLOW_COPY_AND_ASSIGN(IdentityFilter);
};

#endif // IDENTITYFILTER_H_