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
	Net net;
	string caffemodel;
	string prototxt;
	string labels;
	int inputWidth;
	int inputHeight;
	Scalar mean;
	bool swapRB;
	float scale;

public:
	DnnClassificator(string caffemodel1, string prototxt1, string labels1, int inputWidth1, int inputHeight1, Scalar mean1, bool swapRB1, float scale);
	Mat Classify(Mat image);

};

