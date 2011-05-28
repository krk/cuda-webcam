#include "SingleCudaProcessor.h"

#include "cudaCommon.h"

class SingleCudaTexProcessor : public SingleCudaProcessor
{

private:
	cudaArray* cu_array;
	cudaChannelFormatDesc channelDesc;
	const char* textureSymbolName;

public:
	
	SingleCudaTexProcessor( void kernelLauncher(float*, int, int), const char* textureSymbolName )
		: SingleCudaProcessor(kernelLauncher), textureSymbolName(textureSymbolName)
	{
	}

	virtual void InitProcessing(int width, int height);
	virtual void ProcessImage(char* imageData);
	virtual void ReleaseProcessing();
};