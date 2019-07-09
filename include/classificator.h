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
	string pathToModel;
	string pathToConfig;
	string pathToLabels;
	int inputWidth;
	int inputHeight;
	Scalar mean;
	bool swapRB;
	Net net;
public:
	DnnClassificator(string _pathToModel, string _pathToConfig, string _pathToLabels, int _inputWidth, int _inputHeight, Scalar _mean, bool _swapRB);
	Mat Classify(Mat frame);
};