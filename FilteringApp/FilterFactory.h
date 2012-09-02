#pragma once
#ifndef FILTERINGAPP_FILTERFACTORY_H
#define FILTERINGAPP_FILTERFACTORY_H

#include <QtGui/QMainWindow>
#include "..\CudaFilters\FilterFactory.h"

#include <vector>
#include "boost\assert.hpp"

#include "..\CudaFilters\SingleImageFilterChain.h"

typedef ISingleImageFilter* (__stdcall *filterFactoryFunctorType)();

struct FilterFactory{

public:
	 filterFactoryFunctorType Create;	 

	 FilterFactory(const FilterFactory &other){ Create = other.Create; }
	 FilterFactory(filterFactoryFunctorType create){ Create = create; }
	 ~FilterFactory() {}
	 FilterFactory()  { Create = 0; }
	 bool hasConfig(){ return false; }
};

Q_DECLARE_METATYPE(FilterFactory)
Q_DECLARE_METATYPE(FilterFactory*)

enum DataRoles
{
	FilterFactoryData = Qt::UserRole + 2,
	FilterFactoryConfig = Qt::UserRole + 3,
};

#endif // FILTERINGAPP_FILTERFACTORY_H