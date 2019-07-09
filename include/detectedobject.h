#pragma once

#include <string>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

struct DetectedObject
{
	int id;
	float confidence;
	cv::Rect rect;
};

