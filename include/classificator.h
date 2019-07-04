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
class DnnClassificator :  Classificator 
{
	Net net;
	string model;
	string config;
	string labels;
	double inputWidth;
	double inputHeight;
	Scalar mean;
	bool swapRB;
public:
	DnnClassificator::DnnClassificator(string _model, string _config, string _labels, double _inputWidth,double _inputHeight, Scalar _mean, bool Swarp);
	Mat Classify(Mat image);
};