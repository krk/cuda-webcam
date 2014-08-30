#include "GLDualCamView.h"

using namespace cv;

GLDualCamView::GLDualCamView(void)
{
}

GLDualCamView::GLDualCamView(QWidget *parent)
	: CQtOpenCVViewerGl(parent)
{
	isCaptureEnabled = false;
	isProcessingEnabled = false;
	_isFilterInited = false;

	if (!_cap.isOpened())
		if (!_cap.open(0))
		{
			qDebug() << "Kamera tutacagi edinilemedi.";
			return;
		}

	/* Setup the timer. */
	_timer = new QTimer();

	connect(_timer, SIGNAL(timeout()), this, SLOT(captureFrame()));

	_timer->start(33); /* ~30fps */
}

GLDualCamView::~GLDualCamView(void)
{
	if (_filter != 0)
		_filter->ReleaseFilter();

	ReleaseCUDAThread();
}

void GLDualCamView::changeFilter(vector<FilterFactory> filters)
{
	if (_filter != 0)
		_filter->ReleaseFilter();

	/* Save factories for use from GL thread. */
	_filters = filters;

	_isFilterInited = false;
}

void GLDualCamView::captureFrame()
{
	if (!isCaptureEnabled)
	{
		glClear(GL_COLOR_BUFFER_BIT);
		return;
	}

	_cap >> _mat;

	if (_mat.empty())
		return;

	_processedMat = cv::Mat(_mat);

	cv::resize(_mat, _processedMat, cv::Size(640, 480), 0, 0, INTER_CUBIC);

	if (_processedMat.empty())
		return;

	if (isCaptureEnabled
		&& isProcessingEnabled
		&& !_isFilterInited)
	{
		/* BUG: Ayni threadde yaratilmasi icin burada create metodlari cagriliyor, texture iceren filtreler calismiyor! */
		SingleImageFilterChain* chain = new SingleImageFilterChain();
		for (vector<FilterFactory>::iterator it = _filters.begin(); it != _filters.end(); ++it) {
			chain->AppendFilter(it->Create(&(*it)));
		}

		_filter = chain;

		_filter->InitFilter(_processedMat.size().width, _processedMat.size().height, _processedMat.step.buf[0]);
		_isFilterInited = true;
	}

	if (_filter != 0
		&& isCaptureEnabled
		&& isProcessingEnabled)
	{
		_filter->FilterImage((char*) _processedMat.data);
	}

	// draw raw image
	if (isCaptureEnabled && !isProcessingEnabled)
	{
		this->showImage(_mat);

		/*QImage qtFrame(_mat.data, _mat.size().width, _mat.size().height, _mat.step.buf[0], QImage::Format_RGB888);
		qtFrame = qtFrame.rgbSwapped();
		renderImage(qtFrame);*/
	}

	// draw processed image
	if (isCaptureEnabled && isProcessingEnabled)
	{
		this->showImage(_processedMat);

		/*QImage qtFrame(_processedMat.data, _processedMat.size().width, _processedMat.size().height, _processedMat.step.buf[0], QImage::Format_RGB888);
		qtFrame = qtFrame.rgbSwapped();
		renderImage(qtFrame);*/
	}
}
