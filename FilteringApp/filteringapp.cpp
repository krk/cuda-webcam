#include "filteringapp.h"
#include <qstandarditemmodel.h>
#include <QItemSelectionModel>
#include <QMetaType>




FilteringApp::FilteringApp(QWidget *parent, Qt::WFlags flags)
	: QMainWindow(parent, flags)
{
	ui.setupUi(this);

	setupFilterListView();

	connect(
		ui.lvFilters->selectionModel(), 
		SIGNAL(selectionChanged(const QItemSelection&, const QItemSelection&)),
		this, 
		SLOT(filterListSelectionChanged(const QItemSelection&, const QItemSelection&)));	
}
ISingleImageFilter* getFilter();
void FilteringApp::setupFilterListView()
{
	QStandardItemModel *model;

	model = new QStandardItemModel();

	ui.lvFilters->setModel(model);

	QStandardItem *item = new QStandardItem("Test");
	QStandardItem *item2 = new QStandardItem("Test2");

	filterFactoryFunctorType filterFactoryFunctor = &getFilter;
	
	FilterFactory filterFactory;
	filterFactory.Create = filterFactoryFunctor;
	
	QVariant qv = QVariant::fromValue<FilterFactory>(filterFactory);
	
	item->setData(qv, FilterFactoryData);

	model->appendRow(item);
	model->appendRow(item2);
	QItemSelectionModel *qism = ui.lvFilters->selectionModel();	
}

void FilteringApp::filterListSelectionChanged(const QItemSelection &selected, const QItemSelection &deselected)
{
	const QStandardItemModel *item= (const QStandardItemModel *)selected.first().model();
	QVariant data = item->data(selected.first().indexes().first(), FilterFactoryData);
	
	if(data.isNull() || !data.isValid())
		return;

	FilterFactory factory = data.value<FilterFactory>();

	ISingleImageFilter* filter = factory.Create();

	QStandardItemModel *model = ( QStandardItemModel* )ui.lvFilters->model();
	model->appendRow(new QStandardItem("Test4"));
}

ISingleImageFilter* getFilter()
{
	char *d="sds";
	return (ISingleImageFilter*)d;
}

FilteringApp::~FilteringApp()
{

}
