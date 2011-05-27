#include "SingleCudaProcessor.h"

#include "cudaCommon.h"

class SingleCudaTexProcessor : public SingleCudaProcessor
{

private:
	cudaArray* cu_array;
	cudaChannelFormatDesc channelDesc;

public:
	
	SingleCudaTexProcessor( void kernelLauncher(float*, int, int) )
		: SingleCudaProcessor(kernelLauncher)
	{
	}

	virtual void InitProcessing(int width, int height);
	virtual void ProcessImage(char* imageData);
	virtual void ReleaseProcessing();
};