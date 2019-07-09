#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

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
	DnnClassificator(String path_to_model, String path_to_config, String path_to_labels, int inputWidth, int inputHeight, Scalar mean = (0,0,0,0), bool swapRB=false);
	Mat Classify(Mat image);
	~DnnClassificator() {};

private:
	String path_to_model;
	String path_to_config;
	String path_to_labels;
	int inputWidth;
	int inputHeight;
	Scalar mean;
	bool swapRB;
	Net net;
};