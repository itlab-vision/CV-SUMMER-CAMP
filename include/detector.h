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
private:
	string model_path;
	string config_path;
	string labels_path;
	int inputWidth;
	int inputHeight;
	bool swapRB;
	Scalar mean;
	Net net;
public:
	DnnDetector(string path_to_model, string path_to_config, string path_to_labels, int _inputWidth, int _inputHeight, bool _swapRB, Scalar _mean = (0, 0, 0, 0));
	vector <DetectedObject> Detect(Mat image);
};
