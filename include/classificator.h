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

class DnnClassificator :public Classificator {
	string path_to_model, path_to_config, path_to_labels;
	int width, height;
	Scalar mean;
	bool swap;
	Net net;
public:
	DnnClassificator(string ptm, string ptc, string ptl, int nwidth, int nheight, Scalar nmean = (0, 0, 0, 0), bool srb = 0);
	Mat Classify(Mat image);
};