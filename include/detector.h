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


class DnnDetector : public Detector {
private:
	String modelPath;
	String configPath;
	String labelsPath;
	int inputWidth;
	int inputHeight;
	double scale;
	Scalar mean;
	bool swapRB;
	Net net;
public:
	DnnDetector(String _modelPath, String _configPath, String _labelsPath, int _inputWidth, int _inputHeight, double _scale, Scalar _mean, bool _swapRB);
	vector<DetectedObject> Detect(Mat image);
};