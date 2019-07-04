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
	string model_path;
	string config_path;
	string labels_path;
	int inputWidth;
	int inputHeight;
	bool swapRB;
	Scalar mean;
	Net net;
public:
	Mat Classify(Mat image);
	DnnClassificator(string path_to_model, string path_to_config, string path_to_labels, int _inputWidth, int _inputHeight, bool _swapRB, Scalar _mean = (0, 0, 0, 0));
};