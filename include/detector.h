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
private:
	Net net;
	int width, height;
public:
	DnnDetector(String pathToConfig, String pathToModel, int w, int h);
	vector<DetectedObject> Detect(Mat image) override;
};
