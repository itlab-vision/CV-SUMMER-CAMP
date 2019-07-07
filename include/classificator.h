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
	virtual String DecodeLabel(int n) = 0 {}
};

class DnnClassificator : public Classificator {

private:
	String path_to_model;
	String path_to_config;
	String path_to_labels;
	int input_width;
	int input_height;
	Scalar mean;
	bool swapRB;
	Net net;

public:
	DnnClassificator(String path_to_model, String path_to_config, String path_to_labels, int input_width, int input_height, bool swapRB, Scalar mean = (0, 0, 0));
	Mat Classify(Mat image);
	String DecodeLabel(int n);
};