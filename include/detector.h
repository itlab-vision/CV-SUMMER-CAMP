#pragma once
#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "detectedobject.h"
#include "filter.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;

class Detector
{
public:
	virtual vector<DetectedObject> Detect(Mat image, Size size, Scalar mean, bool swapRB, double scale) = 0 {}
};


class DnnDetector : Detector {
public:
	DnnDetector(const String & ptm, const String & ptc, const String& ptl);

	vector<DetectedObject> Detect(Mat image, Size size, Scalar mean, bool swapRB = false, double scale = 1.0);
private:
	Net net;
	std::string ptl;
};