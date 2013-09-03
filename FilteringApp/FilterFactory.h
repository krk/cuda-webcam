#pragma once
#ifndef FILTERINGAPP_FILTERFACTORY_H
#define FILTERINGAPP_FILTERFACTORY_H

#include <QtGui/QMainWindow>
#include "..\CudaFilters\FilterFactory.h"

#include <vector>

#include "..\CudaFilters\SingleImageFilterChain.h"

#include "NumericFilterParameter.h"

class FilterFactory;

typedef ISingleImageFilter* (__stdcall *filterFactoryCreateType)();
typedef ISingleImageFilter* (__stdcall *filterFactoryFunctorType)(FilterFactory*);

class FilterFactory{

public:
	 filterFactoryFunctorType Create;	 	 
	 NumericFilterParameter* Parameter;

	 FilterFactory(const FilterFactory &other){ Create = other.Create; Parameter = other.Parameter; }
	 FilterFactory(filterFactoryFunctorType create, NumericFilterParameter* parameter = NULL)
	 { 
		 Create = create; 
		 Parameter = parameter;
	 }

	 ~FilterFactory() {}
	 FilterFactory()  { Create = 0; Parameter = NULL; }
	 virtual bool hasConfig(){ return Parameter != NULL; }
};

Q_DECLARE_METATYPE(FilterFactory)
Q_DECLARE_METATYPE(FilterFactory*)

enum DataRoles
{
	FilterFactoryData = Qt::UserRole + 2,
	FilterFactoryConfig = Qt::UserRole + 3,
};

#endif // FILTERINGAPP_FILTERFACTORY_H