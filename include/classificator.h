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

class DnnClassificator : Classificator
{
private:
	int width;
	int height;
	string model_path;
	string config_path;
	string labels_path;
	Scalar scalar;
	bool mirror;
	Net net;
public:
	DnnClassificator(string model_path, string config_path, string labels_path, int inputWidth, int inputHeight, bool mirror, Scalar scalar);
	Mat Classify(Mat image);
};