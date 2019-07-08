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

class DnnDetector: public Detector
{
public:
	DnnDetector(String path_to_model, String path_to_config, String path_to_labels, int inputWidth, int inputHeight, Scalar mean = (0, 0, 0, 0), bool swapRB = false);
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
