#pragma once
#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "detectedobject.h"
#include "label_parser.h"
#include "tracking_by_matching.hpp"


using namespace cv;
using namespace cv::dnn;
using namespace std;

class Detector
{

public:
	virtual tbm::TrackedObjects Detector::Detect(Mat image, int frame_idx) = 0;
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
	tbm::TrackedObjects DnnDetector::Detect(Mat image, int frame_idx);
};