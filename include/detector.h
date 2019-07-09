#pragma once
#include <iostream>
#include <string>
#include <fstream>

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
	virtual string DecodeLabel(int n) = 0 {}
};

class DnnDetector : public Detector {
private:
	string path_to_model;
	string path_to_config;
	string path_to_labels;
	int input_width = 300;
	int input_height = 300;
	double scale = 0.007843;
	Scalar mean = Scalar(127.5, 127.5, 127.5);;
	bool swapRB = false;
	Net net;
public:
	DnnDetector(string path_to_model, string path_to_config, string path_to_labels);
	vector<DetectedObject> Detect(Mat image);
	string DecodeLabel(int n);
};
