#ifndef FILTERINGAPP_H
#define FILTERINGAPP_H

#include <QtGui/QMainWindow>
#include "ui_filteringapp.h"
#include "..\CudaFilters\FilterFactory.h"

class FilteringApp : public QMainWindow
{
	Q_OBJECT

public:
	FilteringApp(QWidget *parent = 0, Qt::WFlags flags = 0);
	~FilteringApp();

private:
	Ui::FilteringAppClass ui;
	void setupFilterListView();	
private slots:
	void filterListSelectionChanged(const QItemSelection &selected, const QItemSelection &deselected);
};

enum DataRoles
{
	FilterFactoryData = Qt::UserRole + 2,
};

typedef ISingleImageFilter* (*filterFactoryFunctorType)();

struct FilterFactory{

public:
	 filterFactoryFunctorType Create;	 

	 FilterFactory(const FilterFactory &other){ Create = other.Create; }
	 ~FilterFactory(){}
	 FilterFactory()  {}
};

Q_DECLARE_METATYPE(FilterFactory)
Q_DECLARE_METATYPE(FilterFactory*)

#endif // FILTERINGAPP_H
