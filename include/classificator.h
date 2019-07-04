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
	DnnClassificator(const String& ptm, const String& ptc, uint32_t width, uint32_t height, const Scalar& mean, bool swapRB);

	Mat Classify(Mat image);
private:
	Net net;

	uint32_t width;
	uint32_t height;
	Scalar mean;
	bool swapRB;
};