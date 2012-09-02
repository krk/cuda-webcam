#pragma once
#ifndef FILTERFACTORY_H
#define FILTERFACTORY_H

#include <QtGui/QMainWindow>
#include "..\CudaFilters\FilterFactory.h"

#include <vector>
#include "boost\assert.hpp"
#include "..\CudaFilters\SingleImageFilterChain.h"

struct FilterFactory{

public:
	 filterFactoryFunctorType Create;	 

	 FilterFactory(const FilterFactory &other){ Create = other.Create; }
	 FilterFactory(filterFactoryFunctorType create){ Create = create; }
	 ~FilterFactory() {}
	 FilterFactory()  {}
	 bool hasConfig(){ return false; }
};

Q_DECLARE_METATYPE(FilterFactory)
Q_DECLARE_METATYPE(FilterFactory*)

enum DataRoles
{
	FilterFactoryData = Qt::UserRole + 2,
	FilterFactoryConfig = Qt::UserRole + 3,
};

#endif // FILTERFACTORY_H