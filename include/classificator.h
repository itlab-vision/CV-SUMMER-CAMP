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

class DnnClassificator : virtual Classificator
{
private:
	string pathtoModel;
	string pathtoConfig;
	string pathtoLabels;
	int width;
	int height;
	Scalar mean;
	bool SwapRB;
	Net net;
public:
	DnnClassificator(string newpathtoModel, string newpathtoConfig, string newpathtoLabels, 
		int newWidth, int newHeight, bool newSwapRB)
	{
		pathtoModel = newpathtoModel;
		pathtoConfig = newpathtoConfig;
		pathtoLabels = newpathtoLabels;
		width = newWidth;
		height = newHeight;
		mean = { 103.94,113.78,128 };
		SwapRB = newSwapRB;
		net = readNet(pathtoModel, pathtoConfig);
		net.setPreferableBackend(DNN_BACKEND_OPENCV);
		net.setPreferableTarget(DNN_TARGET_CPU);
	}
	Mat Classify(Mat image);
};