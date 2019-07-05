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
};

class DnnDetector : Detector
{
private:
	int width;
	int height;
	double scale;
	bool mirror;
	string model_path;
	string config_path;
	string labels_path;
	Scalar scalar;
	vector<string> labels;
	Net net;
	
	bool ParseLabels();

public:
	DnnDetector(string model_path, string config_path, string labels_path, int inputWidth, int inputHeight, bool mirror, Scalar scalar = (0, 0, 0), double scale = 1.0);
	vector <DetectedObject> Detect(Mat image);
};