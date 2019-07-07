#pragma once
#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

class Classificator
{
public:
    vector<string> classesNames;
    virtual Mat Classify(Mat image) = 0 {}
};

class DnnClassificator : public Classificator {
private:
	String modelPath;
	String configPath;
	String labelsPath;
	int inputWidth;
	int inputHeight;
	Scalar mean;
	bool swapRB;
	Net net;
public:
	DnnClassificator(String _modelPath, String _configPath, String _labelsPath, int _inputWidth, int _inputHeight, Scalar _mean, bool _swapRB);
	Mat Classify(Mat image);
};