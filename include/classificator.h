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
	String labelsPath;
	int width, height;
	Scalar mean;
	Net net;
	bool swapRB;
public:
	DnnClassificator(String pathToModel, String pathToConfing, String pathToLabels,
		int inputWidth, int inputHeight, Scalar mean, bool swapRB = false);
	Mat Classify(Mat image) override;
};