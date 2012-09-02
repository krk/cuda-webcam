#include "GLDualCamView.h"

GLDualCamView::GLDualCamView(void)
{
}

GLDualCamView::GLDualCamView(QWidget *parent)
	: QGLWidget(parent)
{
	isCaptureEnabled = false;
	isProcessingEnabled = false;
	_isFilterInited = false;
	
	_capture = 0;
	_filter = 0;

	// Ýlk tanýmlý kameranýn tutacaðýný al. (tutacak = handle).
	_capture = cvCaptureFromCAM( 0 ); 
	if( !_capture )
	{
		// Tutacak geçersiz ise programdan çýk.
		qDebug() << "Kamera tutaca?? edinilemedi...\n";
		throw runtime_error("Kamera tutaca?? edinilemedi.");
	}

	_rawFrame = 0;
	_processedFrame = 0;

	/* Setup the timer. */
	_timer = new QTimer();

	connect(_timer, SIGNAL(timeout()), this, SLOT(captureFrame())); 

	_timer->start(33); /* ~30fps */
}

GLDualCamView::~GLDualCamView(void)
{
	if(_capture != 0)
		cvReleaseCapture( &_capture ); // Kameran?n tutaca??n? b?rak?r.

	if(_filter != 0)
		_filter->ReleaseFilter();

	ReleaseCUDAThread();
}

void GLDualCamView::initializeGL()
{
	//Adjust the viewport
	glViewport(0,0,this->width(), this->height());
	
	//Adjust the projection matrix
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-this->width()/2, this->width()/2, this->height()/2, -this->height()/2, -1, 1);	
}

void GLDualCamView::resizeGL(int w, int h)
{
	glViewport(0, 0, (GLint)w, (GLint)h);
}

void GLDualCamView::paintGL()
{
	//Clear the color buffer
	glClear(GL_COLOR_BUFFER_BIT);

	//Set the raster position	
	glRasterPos2i(this->width()/2,-this->height()/2);
	//Inver the image (the data coming from OpenCV is inverted)
	glPixelZoom(-1.0f,-1.0f);

	
	// draw raw image
	if(_rawFrame != 0 && isCaptureEnabled && !isProcessingEnabled)
		glDrawPixels(_rawFrame->width, _rawFrame->height, GL_RGB, GL_UNSIGNED_BYTE, _rawFrame->imageData);		
		
	//Draw image from OpenCV capture
	if(_processedFrame != 0 && isCaptureEnabled && isProcessingEnabled)
		glDrawPixels(_processedFrame->width, _processedFrame->height, GL_RGB, GL_UNSIGNED_BYTE, _processedFrame->imageData);		
}

void GLDualCamView::changeFilter(vector<FilterFactory> filters)
{	
	if(_filter != 0)
		_filter->ReleaseFilter();

	/* Save factories for use from GL thread. */
	_filters = filters;	

	_isFilterInited = false;
}

void GLDualCamView::captureFrame()
{
	if(!isCaptureEnabled)
	{
		glClear(GL_COLOR_BUFFER_BIT);		
		return;
	}

	_rawFrame = cvQueryFrame( _capture ); 

	if( !_rawFrame )
	{
		qDebug() << "Kare yakalanamad?.";
		throw runtime_error("Kare yakalanamad?.");
	}
	
	if(_processedFrame == 0)
		_processedFrame = cvCreateImage(cvSize(640, 480), _rawFrame->depth, _rawFrame->nChannels);

	cvResize( _rawFrame, _processedFrame );

	if( !_processedFrame )
		throw runtime_error("Cannot resize image.");
	
	if(isCaptureEnabled 
		&& isProcessingEnabled 
		&& !_isFilterInited)	
	{			
		/* BUG: Ayni threadde yaratilmasi icin burada create metodlari cagriliyor, texture iceren filtreler calismiyor! */
		SingleImageFilterChain* chain = new SingleImageFilterChain();
		for(vector<FilterFactory>::iterator it = _filters.begin(); it != _filters.end(); ++it) {
			chain->AppendFilter(it->Create());
		}

		_filter = chain;

		_filter->InitFilter(_processedFrame->width, _processedFrame->height, _processedFrame->widthStep);
		_isFilterInited = true;
	}

	if(_filter != 0 
		&& isCaptureEnabled 
		&& isProcessingEnabled)
	{
		_filter->FilterImage(_processedFrame->imageData);
	}

	updateGL();
}