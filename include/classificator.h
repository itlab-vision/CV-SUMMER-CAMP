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

class DnnClassificator:Classificator
{
private:
	Net net;
	int width;
	int height;
	Scalar mean;
	bool swapRB;
public:
	DnnClassificator(string path_to_model, string path_to_config, int inputWidth, int inputHeight, Scalar mean, bool swapRG);
	Mat Classify(Mat image) override;
};