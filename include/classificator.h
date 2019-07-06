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

class DnnClassificator :Classificator
{
private:
	String path_to_model;
	String path_to_config;
	String path_to_lables;
	Size spatial_size;
	Scalar mean;
	bool swapRB;
	Net net;
public:
	DnnClassificator(String path_to_model,
		String path_to_config,
		String path_to_lables,
		int width,
		int height,
		Scalar mean,
		bool swapRB);
	Mat Classify(Mat image);
};