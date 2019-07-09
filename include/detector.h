#pragma once
#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2\core\mat.inl.hpp>

#include "detectedobject.h"
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace std;

class Detector
{
public:
    virtual vector<DetectedObject> Detect(Mat image) = 0 {}
};

class DnnDetector: public Detector
{
public:
	DnnDetector();
	vector<DetectedObject> Detect(Mat image);
	~DnnDetector() {};

private:
	String path_to_model;
	String path_to_config;
	String path_to_labels;
	int inputWidth;
	int inputHeight;
	Scalar mean;
	bool swapRB;
	Net net;
};
