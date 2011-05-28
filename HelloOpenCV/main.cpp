#include "main.h" // tüm baþlýklarýmýzý ve tanýmlamalarýmýzý içeren baþlýk dosyasý.



int main( int argc, char** argv )
{
	int key = -1;

	CvCapture* capture = 0;
	CvCapture* capture2 = 0;

	// Ýlk tanýmlý kameranýn tutacaðýný al. (tutacak = handle).
	capture = cvCaptureFromCAM( 0 ); 
	if( !capture )
	{
		// Tutacak geçersiz ise programdan çýk.
		fprintf( stderr, "Kamera tutacaðý edinilemedi...\n" ); // bilgilendiiir.
		exit( EXIT_FAILURE );
	}

	// Görüntü için pointer.
	IplImage* videoFrame = NULL; 
	
	// ilk kareyi yakala.
	videoFrame = cvQueryFrame( capture ); 

	if( !videoFrame )
	{
		// Kare yakalanamadý, programdan çýk.
		printf( "Kare yakalanamadý...\n" );
		exit( EXIT_FAILURE );
	}
	
	// MainVideo adýnda bir gösterim penceresi yarat.
	cvNamedWindow( "MainVideo", 1 );
	
	IplImage* resizedImage = cvCreateImage(cvSize(640, 480), videoFrame->depth, videoFrame->nChannels);

	//ISingleImageFilter* myFilter1 = new SingleCudaFilter(deviceInvertLaunch);
	ISingleImageFilter* myFilter2 = new CpuInvertFilter();
	//ISingleImageFilter* myFilter3 = new SingleCudaFilter(deviceTileFlipLaunch);
	//ISingleImageFilter* myFilter4 = new SingleCudaTexFilter(deviceTexBoxBlurLaunch, "texBlur1");
	ISingleImageFilter* myFilter5 = new SingleCudaTexFilter(deviceTexAbsDiffLaunch, "texAbsDiff1");
	ISingleImageFilter* myFilter6 = new SingleCudaTexFilter(deviceTexInvertLaunch, "tex");
	
	SingleImageFilterChain* myFilter = new SingleImageFilterChain();
	myFilter->AppendFilter( myFilter5 );
	myFilter->AppendFilter( myFilter2 );
	//myFilter->AppendFilter( myFilter6 );

	myFilter->InitFilter(resizedImage->width, resizedImage->height);
	
	// q tuþuna basana kadar dön.
	while( key != 'q' )
	{
		// Bir kare al.
		videoFrame = cvQueryFrame( capture );

		cvResize( videoFrame, resizedImage );

		// Kare geçersiz ise loop biter.
		if( !resizedImage )
			break;

		myFilter->FilterImage(resizedImage->imageData);

		// Negatif görüntüyü pencerede göster.
		cvShowImage( "MainVideo", resizedImage );

		key = cvWaitKey( 10 ); // 10ms tuþ için bekle.

	}

	cvDestroyAllWindows(); // yarattýðýmýz pencereyi yokeder.

	cvReleaseCapture( &capture ); // Kameranýn tutacaðýný býrakýr.

	myFilter->ReleaseFilter();

	cudaThreadExit();

	//exit( EXIT_SUCCESS );
}