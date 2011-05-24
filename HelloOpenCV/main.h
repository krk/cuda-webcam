#pragma once

#include <stdio.h>
#include <stdlib.h>

#include "opencv2\opencv.hpp"

#include "cudaCommon.h"

#include "InvertProcessor.h"

void ProcessFrame(IplImage* videoFrame);