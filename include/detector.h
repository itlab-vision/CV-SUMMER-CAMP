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

class DnnDetector :public Detector
{private:
	String path_to_model;
	String path_to_config;
	String path_to_lables;
	Size spatial_size=Size(300,300);
	double scale = 0.00784;
	Scalar mean=Scalar(127.5, 127.5, 127.5);
	bool swapRB=false;
	Net net;
public:
	DnnDetector(String path_to_model, String path_to_config, String path_to_lables);
	vector<DetectedObject> Detect(Mat image);
};