
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
    virtual Mat Classify(Mat image) = 0;
};
class DnnClassificator : public Classificator {
private:
    Net net;
    string model;
    string config;
    string labels;
    double scale;
    float inWidth;
    float inHeight;
    Scalar mean;
    bool swapRB;

public:
    DnnClassificator(string model, string config, string labels, double scale, float inWidth, float inHeight, Scalar mean, bool swapRB);
    Mat Classify(Mat image);
};