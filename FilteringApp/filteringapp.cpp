#include "filteringapp.h"

FilteringApp::FilteringApp(QWidget *parent, Qt::WindowFlags flags)
	: QMainWindow(parent, flags)
{
	ui.setupUi(this);

	ui.grpParameters->setVisible(false);

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

	connect(
		ui.actionCapture,
		SIGNAL(triggered(bool)),
		this,
		SLOT(actionCapture_triggered(bool)));

	connect(
		ui.actionProcess,
		SIGNAL(triggered(bool)),
		this,
		SLOT(actionProcess_triggered(bool)));	

	connect(this,
		SIGNAL(filterChanged()),
		this,
		SLOT(changeFilter()));

	connect(
		ui.spinParameter,
		SIGNAL(valueChanged(int)),
		this,
		SLOT(spinParameter_changed(int)));	
}

/* Start Currying filters. */

ISingleImageFilter* __stdcall CpuMovingAverageFilter(FilterFactory* factory)
{
	if(factory->hasConfig())
	{
		char param = factory->Parameter->getValue();
		return GetCpuMovingAverageFilter((int)param);
	}
	else	
		return GetCpuMovingAverageFilter(10);	
}

ISingleImageFilter* __stdcall ThresholdFilterFromFactory(FilterFactory* factory)
{
	if(factory->hasConfig())
	{
		unsigned char param = factory->Parameter->getValue();
		return GetThresholdFilter(param);
	}
	else	
		return GetThresholdFilter(192);
}

ISingleImageFilter* __stdcall GetCpuInvertFilterFromFactory(FilterFactory* factory)
{
	return GetCpuInvertFilter();
}

ISingleImageFilter* __stdcall GetCudaInvertFilterFromFactory(FilterFactory* factory)
{
	return GetCudaInvertFilter();
}

ISingleImageFilter* __stdcall GetCudaSepiaFilterFromFactory(FilterFactory* factory)
{
	return GetCudaSepiaFilter();
}

ISingleImageFilter* __stdcall GetCpuCCLFilterFromFactory(FilterFactory* factory)
{
	return GetCpuCCLFilter();
}

ISingleImageFilter* __stdcall GetCudaTexBoxBlurFilterFromFactory(FilterFactory* factory)
{
	return GetCudaTexBoxBlurFilter();
}

ISingleImageFilter* __stdcall GetCudaTexInvertFilterFromFactory(FilterFactory* factory)
{
	return GetCudaTexInvertFilter();
}

ISingleImageFilter* __stdcall GetCudaTileFlipFilterFromFactory(FilterFactory* factory)
{
	return GetCudaTileFlipFilter();
}

ISingleImageFilter* __stdcall GetIdentityFilterFromFactory(FilterFactory* factory)
{
	return GetIdentityFilter();
}

ISingleImageFilter* __stdcall GetAmpInvertFilterFromFactory(FilterFactory* factory)
{
	return GetAmpInvertFilter();
}

/* End Currying */

void FilteringApp::setupFilterListCombo()
{
	QStandardItemModel *model;
	model = new QStandardItemModel();

	ui.cmbFilterType->setModel(model);

	QStandardItem *item = getFilterItem("CPU Invert Filter", &GetCpuInvertFilterFromFactory);
	model->appendRow(item);

	item = getFilterItem("CPU Connected Components Labeler", &GetCpuCCLFilterFromFactory);
	model->appendRow(item);	

	item = getFilterItem("CPU Moving Average", &CpuMovingAverageFilter, new NumericFilterParameter(0, 255, "Moving Average Frames"));
	model->appendRow(item);	

	item = getFilterItem("CUDA Invert", &GetCudaInvertFilterFromFactory);
	model->appendRow(item);

	item = getFilterItem("CUDA Sepia", &GetCudaSepiaFilterFromFactory);
	model->appendRow(item);

	item = getFilterItem("CUDA Texture Box Blur", &GetCudaTexBoxBlurFilterFromFactory);
	model->appendRow(item);

	item = getFilterItem("CUDA Texture Invert", &GetCudaTexInvertFilterFromFactory);
	model->appendRow(item);

	item = getFilterItem("CUDA Tile Flip", &GetCudaTileFlipFilterFromFactory);
	model->appendRow(item);

	item = getFilterItem("Identity", &GetIdentityFilterFromFactory);
	model->appendRow(item);	

	item = getFilterItem("Threshold", &ThresholdFilterFromFactory, new NumericFilterParameter(0, 255, "Threshold"));
	model->appendRow(item);	

	item = getFilterItem("C++ AMP Invert", &GetAmpInvertFilterFromFactory);
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
	return getFilterItem(text, factory, NULL);
}

QStandardItem* FilteringApp::getFilterItem(QString text, filterFactoryFunctorType factory, NumericFilterParameter* parameter)
{
	QStandardItem *item = new QStandardItem(text);

	FilterFactory filterFactory(factory, parameter);		

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
		setupParameters(&factory);
	}
	else
	{
		setupParameters(NULL);
	}
}

void FilteringApp::setupParameters(FilterFactory* factory)
{
	if(factory == NULL || factory->Parameter == NULL)
	{		
		ui.grpParameters->setVisible(false);
		return;
	}

	QString caption = factory->Parameter->getCaption();
	
	ui.lblParameterCaption->setText(caption);
	ui.spinParameter->setValue((int)factory->Parameter->getValue());	

	ui.grpParameters->setVisible(true);
}

void FilteringApp::cmbFiltersIndexChanged(int index)
{	
	
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
	QStandardItem *listItem = getFilterItem(text, factory.Create, factory.Parameter);

	/* Add new item to the list. */
	QStandardItemModel *lvModel = (QStandardItemModel *)ui.lvFilters->model();
	lvModel->appendRow(listItem);

	emit filterChanged();
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

	emit filterChanged();
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

	emit filterChanged();
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

	emit filterChanged();
}

vector<FilterFactory> FilteringApp::GetFilters()
{
	vector<FilterFactory> ret;

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

		/* Get FilterFactory object. */
		FilterFactory factory = data.value<FilterFactory>();

		/* Push factory to return vector. */
		ret.push_back(factory);
	}

	return ret;
}

void FilteringApp::actionCapture_triggered(bool checked)
{
	ui.camDual->isCaptureEnabled = checked;
}

void FilteringApp::actionProcess_triggered(bool checked)
{
	ui.camDual->isProcessingEnabled = checked;
}

void FilteringApp::spinParameter_changed(int value)
{	
	if(ui.lvFilters->selectionModel()->selectedIndexes().count() == 0)
		return;

	QModelIndexList indexes = ui.lvFilters->selectionModel()->selectedIndexes();
	QVariant data = indexes.first().data(FilterFactoryData);

	if(data.isNull() || !data.isValid())
		return;

	FilterFactory factory = data.value<FilterFactory>();

	if(!factory.hasConfig())
		return;

	if(factory.Parameter == NULL)
		return;

	factory.Parameter->setValue((char)value);

	emit filterChanged();
}

void FilteringApp::changeFilter()
{	
	vector<FilterFactory> filters = GetFilters();

	ui.camDual->changeFilter(filters);
}

FilteringApp::~FilteringApp()
{

}

#if _MSC_VER < 1700 
/*  Switch toolset to v110, include AmpFilters as a reference and rebuild to use AmpFilters. 
Might need Windows SDK ver. 8 for opengl32.lib and friends */
ISingleImageFilter* __stdcall GetAmpInvertFilter()
{
	return GetCpuInvertFilter();
}
#endif