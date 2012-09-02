#include "filteringapp.h"
#include <qstandarditemmodel.h>
#include <QItemSelectionModel>
#include <QMetaType>
#include "FilterFactory.h"

FilteringApp::FilteringApp(QWidget *parent, Qt::WFlags flags)
	: QMainWindow(parent, flags)
{
	ui.setupUi(this);

	setupFilterListCombo();
	setupFilterListView();

	connect(
		ui.lvFilters->selectionModel(), 
		SIGNAL(selectionChanged(const QItemSelection&, const QItemSelection&)),
		this, 
		SLOT(filterListSelectionChanged(const QItemSelection&, const QItemSelection&)));	
	
	connect(
		ui.cmbFilterType, 
		SIGNAL(currentIndexChanged(int)),
		this, 
		SLOT(cmbFiltersIndexChanged(int)));	

	connect(
		ui.pbAddFilter, 
		SIGNAL(clicked()),
		this, 
		SLOT(pbAddFilter_clicked()));	

	connect(
		ui.pbRemove, 
		SIGNAL(clicked()),
		this, 
		SLOT(pbRemoveFilter_clicked()));	

	connect(
		ui.pbMoveFilterUp, 
		SIGNAL(clicked()),
		this, 
		SLOT(pbMoveFilterUp_clicked()));	

	connect(
		ui.pbMoveFilterDown, 
		SIGNAL(clicked()),
		this, 
		SLOT(pbMoveFilterDown_clicked()));	
}

ISingleImageFilter* __stdcall CpuMovingAverageFilter()
{
	return GetCpuMovingAverageFilter(10);
}

ISingleImageFilter* __stdcall ThresholdFilter()
{
	return GetThresholdFilter(192);
}

void FilteringApp::setupFilterListCombo()
{
	QStandardItemModel *model;
	model = new QStandardItemModel();

	ui.cmbFilterType->setModel(model);

	QStandardItem *item = getFilterItem("CPU Invert Filter", &GetCpuInvertFilter);
	model->appendRow(item);

	item = getFilterItem("CPU Connected Components Labeler", &GetCpuCCLFilter);
	model->appendRow(item);	

	item = getFilterItem("CPU Moving Average", &CpuMovingAverageFilter);
	model->appendRow(item);	

	item = getFilterItem("CUDA Invert", &GetCudaInvertFilter);
	model->appendRow(item);

	item = getFilterItem("CUDA Sepia", &GetCudaSepiaFilter);
	model->appendRow(item);

	item = getFilterItem("CUDA Texture Box Blur", &GetCudaTexBoxBlurFilter);
	model->appendRow(item);

	item = getFilterItem("CUDA Texture Invert", &GetCudaTexInvertFilter);
	model->appendRow(item);

	item = getFilterItem("CUDA Tile Flip", &GetCudaTileFlipFilter);
	model->appendRow(item);

	item = getFilterItem("Identity", &GetIdentityFilter);
	model->appendRow(item);	

	item = getFilterItem("Threshold", &ThresholdFilter);
	model->appendRow(item);	
}

void FilteringApp::setupFilterListView()
{
	QStandardItemModel *model;

	model = new QStandardItemModel();

	ui.lvFilters->setModel(model);
}

QStandardItem* FilteringApp::getFilterItem(QString text, filterFactoryFunctorType factory)
{
	QStandardItem *item = new QStandardItem(text);
	
	FilterFactory filterFactory(factory);		
	QVariant qv = QVariant::fromValue<FilterFactory>(filterFactory);		
	item->setData(qv, FilterFactoryData);	
	return item;
}

void FilteringApp::filterListSelectionChanged(const QItemSelection &selected, const QItemSelection &deselected)
{
	if(selected.count() == 0)
		return;

	const QStandardItemModel *item= (const QStandardItemModel *)selected.first().model();
	QVariant data = item->data(selected.first().indexes().first(), FilterFactoryData);
	
	if(data.isNull() || !data.isValid())
		return;

	FilterFactory factory = data.value<FilterFactory>();

	if(factory.hasConfig())
	{
		/* Confige uygun arayüz oluştur. */
	}
}

void FilteringApp::cmbFiltersIndexChanged(int index)
{	
	QVariant data = ui.cmbFilterType->model()->index(index, 0).data(FilterFactoryData);

	if(data.isNull() || !data.isValid())
		return;

	FilterFactory factory = data.value<FilterFactory>();
}

void FilteringApp::pbAddFilter_clicked()
{
	/* Get combobox model. */
	QStandardItemModel *cmbModel = (QStandardItemModel *)ui.cmbFilterType->model();

	/* Find selected item. */
	int index = ui.cmbFilterType->currentIndex();
	QStandardItem *item = cmbModel->item(index);
	
	/* Get filter factory data. */
	QVariant data = item->data(FilterFactoryData);
	
	if(data.isNull() || !data.isValid())
		return;

	FilterFactory factory = data.value<FilterFactory>();
	QString text = item->text();
	
	/* Create a new item for the list. */
	QStandardItem *listItem = getFilterItem(text, factory.Create);

	/* Add new item to the list. */
	QStandardItemModel *lvModel = (QStandardItemModel *)ui.lvFilters->model();
	lvModel->appendRow(listItem);
}

void FilteringApp::pbRemoveFilter_clicked()
{
	/* Get listview model. */
	QStandardItemModel *lvModel = (QStandardItemModel *)ui.lvFilters->model();

	/* Find selected item. */
	QModelIndex index = ui.lvFilters->currentIndex();
	
	if(!index.isValid())
		return;

	/* Remove item from model. */
	lvModel->removeRow(index.row());
}


void FilteringApp::pbMoveFilterUp_clicked()
{
	/* Get listview model. */
	QStandardItemModel *lvModel = (QStandardItemModel *)ui.lvFilters->model();

	/* Find selected item. */
	QModelIndex index = ui.lvFilters->currentIndex();
	
	if(!index.isValid() 
		|| index.row() == 0)
		return;
	
	/* Get item from index. */
	QStandardItem *item = lvModel->takeItem(index.row());

	/* Remove row. */
	lvModel->removeRow(index.row());

	/* Insert row above. */
	lvModel->insertRow(index.row() - 1, item);
	
	/* Get new index. */
	QModelIndex newIndex = lvModel->indexFromItem(item);

	/* Select new index. */
	ui.lvFilters->setCurrentIndex(newIndex);
}

void FilteringApp::pbMoveFilterDown_clicked()
{
	/* Get listview model. */
	QStandardItemModel *lvModel = (QStandardItemModel *)ui.lvFilters->model();

	/* Find selected item. */
	QModelIndex index = ui.lvFilters->currentIndex();
	
	if(!index.isValid() 
		|| index.row() == lvModel->rowCount() - 1)
		return;
	
	/* Get item from index. */
	QStandardItem *item = lvModel->takeItem(index.row());

	/* Remove row. */
	lvModel->removeRow(index.row());

	/* Insert row below. */
	lvModel->insertRow(index.row() + 1, item);
	
	/* Get new index. */
	QModelIndex newIndex = lvModel->indexFromItem(item);

	/* Select new index. */
	ui.lvFilters->setCurrentIndex(newIndex);
}

SingleImageFilterChain* FilteringApp::GetFilterChain()
{
	/* Get listview model. */
	QStandardItemModel *lvModel = (QStandardItemModel *)ui.lvFilters->model();

	/* Create new chain */
	SingleImageFilterChain *chain = new SingleImageFilterChain();

	/* iterate list items. */
	for(int i = 0; i < lvModel->rowCount(); ++i)
	{
		/* Get current item. */
		QStandardItem *item = lvModel->item(i);

		/* Get filter factory data. */
		QVariant data = item->data(FilterFactoryData);
	
		if(data.isNull() || !data.isValid())
			continue;

		FilterFactory factory = data.value<FilterFactory>();
		ISingleImageFilter *filter = factory.Create();

		chain->AppendFilter(filter);
	}

	return chain;
}

FilteringApp::~FilteringApp()
{

}
