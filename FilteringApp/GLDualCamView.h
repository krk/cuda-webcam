#pragma once

#ifndef GLDUALCAMVIEW_H
#define GLDUALCAMVIEW_H
#include <QObject>
#include <QGLWidget>
#include "opencv2\opencv.hpp"
#include <QtDebug>
#include <QTimer>
#include "..\CudaFilters\ISingleImageFilter.h"
#include "FilterFactory.h"
#include <QtOpenGL>
#include <qopengl.h>
#include "CQtOpenCVViewerGl.h"

class GLDualCamView :
	public CQtOpenCVViewerGl
{
	Q_OBJECT
public:
	GLDualCamView(void);
	~GLDualCamView(void);
	GLDualCamView(QWidget *parent);

	bool isCaptureEnabled;
	bool isProcessingEnabled;
		
	void changeFilter(vector<FilterFactory> filters);
	
private slots:
	 void captureFrame();	 

private:
	 cv::VideoCapture _cap;
	 cv::Mat _mat;
	 cv::Mat _processedMat;

	 QTimer* _timer;

	 bool _isFilterInited;	 
	 ISingleImageFilter* _filter;	
	 vector<FilterFactory> _filters;
};

#endif // GLDUALCAMVIEW_H