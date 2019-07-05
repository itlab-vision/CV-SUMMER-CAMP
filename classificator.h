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

class DnnClassificator : public Classificator
{
public:
	string modelPath, configPath, labelsPath;
	int width, height;
	bool swapRB;
	Scalar mean;
	Net net;
	DnnClassificator(string modelPath_, string configPath_, string labelsPath_,	int inputWidth_, int inputHeight_, Scalar mean_ = (0, 0, 0, 0), bool swapRB_ = false);
	Mat Classify(Mat image);
};