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

class DnnDetector : Detector
{
	Net net;
	string path_to_model;
	string path_to_confing;
	string path_to_labels;

public:
	DnnDetector(string _path_to_model, string _path_to_confing, string _path_to_labels);
	vector<DetectedObject> Detect(Mat image) override;
};
