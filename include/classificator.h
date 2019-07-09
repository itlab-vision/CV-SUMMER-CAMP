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


class DnnClassificator : Classificator {
public:
	DnnClassificator(string model_path, string config_path, int inputWidth, int inputHeight, bool swapRB, Scalar scalar);
	Mat Classify(Mat image);
private:
	Net net;
	int width;
	int height;
	bool swapRB;
	Scalar scalar;
};