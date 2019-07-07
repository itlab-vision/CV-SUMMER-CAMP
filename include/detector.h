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
	Net net;
	string model;
	string config;
	string labels;
	int inputWidth;
	int inputHeight;
	Scalar mean;
	double scale;
	bool swapRB;
public:
	DnnDetector(string _model, 
				string _config, 
				string _labels,
				int _inputWidth, 
				int _inputHeight, 
				Scalar _mean,
				double _scale,
				bool _swapRB);
	vector <DetectedObject> Detect(Mat image);

};
