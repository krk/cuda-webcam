#pragma once
#ifndef FILTERINGAPP_H
#define FILTERINGAPP_H

#include <QtGui/QMainWindow>
#include "ui_filteringapp.h"
#include "..\CudaFilters\FilterFactory.h"
#include <QString>
#include <QStandardItem>
#include "..\CudaFilters\SingleImageFilterChain.h"

typedef ISingleImageFilter* (__stdcall *filterFactoryFunctorType)();

class FilteringApp : public QMainWindow
{
	Q_OBJECT

public:
	FilteringApp(QWidget *parent = 0, Qt::WFlags flags = 0);
	~FilteringApp();

private:
	Ui::FilteringAppClass ui;
	void setupFilterListView();	
	void setupFilterListCombo();
	QStandardItem* getFilterItem(QString text, filterFactoryFunctorType factory);	
	SingleImageFilterChain* GetFilterChain();

private slots:
	void filterListSelectionChanged(const QItemSelection &selected, const QItemSelection &deselected);
	void cmbFiltersIndexChanged(int index);
	void pbAddFilter_clicked();
	void pbRemoveFilter_clicked();
	void pbMoveFilterUp_clicked();
	void pbMoveFilterDown_clicked();
};

#endif // FILTERINGAPP_H
