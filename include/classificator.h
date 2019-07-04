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

class DnnClassificator : public Classificator {
private:
	string Tomodel;
	string Toconfig; 
	string Tolabels; 
	float inputWidth; 
	float inputHeight; 
	Scalar mean; 
	bool swapRB;
	double scale;

public:
	DnnClassificator(string Tomodel1, string Toconfig1, string Tolabels1, double scale1, float inputWidth1, float  inputHeight1, Scalar mean1, bool swapRB1);
	Mat Classify(Mat image);
};