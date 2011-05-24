#pragma once

class ISingleImageProcessor
{
public:

	// Ýmaj ile ilgili alloc ve benzeri iþlemleri yapar.
	virtual void InitProcessing(int width, int height) = 0;

	// Ýmajý yerinde iþleyen fonksiyon (in-place).
	virtual void ProcessImage(char* image) = 0;

	virtual void ReleaseProcessing() = 0;
};

class SingleImageProcessor : public ISingleImageProcessor
{
protected:
	bool isInited;
	bool isReleased;
	int height;
	int width;

public:

	virtual ~SingleImageProcessor()
	{
	}

	SingleImageProcessor(void)
		: isInited(false), isReleased(false), height(0), width(0)
	{
		
	}

	virtual void InitProcessing(int width, int height)
	{
		this->width = width;
		this->height = height;
		this->isInited = true;
	}

	virtual void ReleaseProcessing()
	{
		this->isReleased = true;
	}
};

