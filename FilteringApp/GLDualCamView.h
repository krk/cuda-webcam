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

class GLDualCamView :
	public QGLWidget
{
	Q_OBJECT
public:
	GLDualCamView(void);
	~GLDualCamView(void);
	GLDualCamView(QWidget *parent);

	bool isCaptureEnabled;
	bool isProcessingEnabled;
		
	void changeFilter(vector<FilterFactory> filters);

protected:
     void initializeGL();
	 void resizeGL(int w, int h);
	 void paintGL();	 

private slots:
	 void captureFrame();	 

private:
	 CvCapture* _capture;
	 IplImage* _rawFrame;
	 IplImage* _processedFrame;
	 QTimer* _timer;
	 bool _isFilterInited;	 
	 ISingleImageFilter* _filter;	
	 vector<FilterFactory> _filters;
};

#endif // GLDUALCAMVIEW_H