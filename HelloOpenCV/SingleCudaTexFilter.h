#include "SingleCudaFilter.h"

#include "cudaCommon.h"

class SingleCudaTexFilter : public SingleCudaFilter
{

private:
	cudaArray* cu_array;
	cudaChannelFormatDesc channelDesc;
	const char* textureSymbolName;

	const textureReference* constTexRefPtr;
	textureReference* texRefPtr;

public:
	
	explicit SingleCudaTexFilter( void kernelLauncher(float*, int, int), const char* textureSymbolName )
		: SingleCudaFilter(kernelLauncher), 
		textureSymbolName(textureSymbolName),
		constTexRefPtr(NULL),
		texRefPtr(NULL)
	{
	}

	virtual void InitFilter(int width, int height);
	virtual void FilterImage(char* imageData);
	virtual void ReleaseFilter();
};