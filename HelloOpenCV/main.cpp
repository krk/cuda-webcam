// Copyright (c) 2011 Kerem KAT 
// 
// http://dissipatedheat.com/
// Do not hesisate to contact me about usage of the code or to make comments 
// about the code. Your feedback will be appreciated.
// keremkat<@>gmail<.>com
//
// Kodun kullanýmý hakkýnda veya yorum yapmak için benimle iletiþim kurmaktan
// çekinmeyiniz. Geri bildirimleriniz deðerlendirilecektir.
// keremkat<@>gmail<.>com
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to 
// deal in the Software without restriction, including without limitation the 
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
// sell copies of the Software, and to permit persons to whom the Software is 
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

/**
	\file main.cpp
	Programýn baþlangýç metodunu içeren dosya.
*/


#include "main.h" // tüm baþlýklarýmýzý ve tanýmlamalarýmýzý içeren baþlýk dosyasý.


/**
	Main metod.
*/
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

	//ISingleImageFilter* myFilter0 = new IdentityFilter();
	
	//ISingleImageFilter* myFilter1 = new SingleCudaFilter(deviceInvertLaunch);
	//ISingleImageFilter* myFilter1 = new CudaInvertFilter();

	ISingleImageFilter* myFilter2 = new CpuInvertFilter();
	
	//ISingleImageFilter* myFilter3 = new SingleCudaFilter(deviceTileFlipLaunch);
	//ISingleImageFilter* myFilter3 = new CudaTileFlipFilter();

	//ISingleImageFilter* myFilter4 = new SingleCudaTexFilter(deviceTexBoxBlurLaunch, "texBlur1");
	ISingleImageFilter* myFilter4 = new CudaTexBoxBlurFilter();

	ISingleImageFilter* myFilter5 = new SingleCudaTexFilter(deviceTexAbsDiffLaunch, "texAbsDiff1");

	//ISingleImageFilter* myFilter6 = new SingleCudaTexFilter(deviceTexInvertLaunch, "texInvert1");
	//ISingleImageFilter* myFilter6 = new CudaTexInvertFilter();

	//ISingleImageFilter* myFilter7 = new CudaSepiaFilter();
	ISingleImageFilter* myFilter8 = new CpuCCLFilter();

	ISingleImageFilter* myFilter9 = new ThresholdFilter(90);

	SingleImageFilterChain* myFilter = new SingleImageFilterChain();
	myFilter->AppendFilter( myFilter9 );
	myFilter->AppendFilter( myFilter2 );
	myFilter->AppendFilter( myFilter8 );

	myFilter->InitFilter(resizedImage->width, resizedImage->height, resizedImage->widthStep);
	
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
		cvShowImage( "Unmodified", videoFrame );

		key = cvWaitKey( 10 ); // 10ms tuþ için bekle.

	}

	cvDestroyAllWindows(); // yarattýðýmýz pencereyi yokeder.

	cvReleaseCapture( &capture ); // Kameranýn tutacaðýný býrakýr.

	myFilter->ReleaseFilter();
	
	delete myFilter;

	cudaThreadExit();

	//exit( EXIT_SUCCESS );
}