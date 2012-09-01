#include "filteringapp.h"
#include <QtGui/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	FilteringApp w;
	w.show();
	return a.exec();
}
