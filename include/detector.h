#pragma once
#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "detectedobject.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;

class Detector
{
public:
    virtual vector<DetectedObject> Detect(Mat image) = 0 {}
};

class DnnDetector : public Detector
{
public:
	string modelPath, configPath, labelsPath;
	vector<string> labels;
	int width, height, numObj;
	bool swapRB;
	double scale;
	Scalar mean;
	Net net;

	
	DnnDetector(string _modelPath, string _configPath, string _labelsPath,
		int inputWidth, int inputHeight, Scalar _mean = (0, 0, 0, 0), bool _swapRB = false, double scale = 1.0);

	vector<DetectedObject> Detect(Mat image);
};