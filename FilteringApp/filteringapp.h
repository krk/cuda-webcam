#pragma once
#ifndef FILTERINGAPP_H
#define FILTERINGAPP_H

#include <QtGui/QMainWindow>
#include "ui_filteringapp.h"
#include "..\CudaFilters\FilterFactory.h"
#include "..\AmpFilters\FilterFactory.h"

#include <QString>
#include <QStandardItem>
#include "..\CudaFilters\SingleImageFilterChain.h"

#include <qstandarditemmodel.h>
#include <QItemSelectionModel>
#include <QMetaType>
#include "opencv2\opencv.hpp"
#include "FilterFactory.h"

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
	vector<FilterFactory> GetFilters();

private slots:
	void filterListSelectionChanged(const QItemSelection &selected, const QItemSelection &deselected);
	void cmbFiltersIndexChanged(int index);
	void pbAddFilter_clicked();
	void pbRemoveFilter_clicked();
	void pbMoveFilterUp_clicked();
	void pbMoveFilterDown_clicked();
	void actionCapture_triggered(bool checked);
	void actionProcess_triggered(bool checked);
	void changeFilter();

signals:
	void filterChanged();	
};

#endif // FILTERINGAPP_H
