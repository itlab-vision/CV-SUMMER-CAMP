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
    String modelPath;
    String configPath;
    String labelsPath;
    int inputWidth;
    int inputHeight;
    Scalar mean;
    bool swapRB;
    Net net;
public:
    DnnClassificator(String modelPath, String configPath, String labelsPath, int inputWidth, int inputHeight, Scalar mean = (0, 0, 0, 0), bool swapRB = false);
    Mat Classify(Mat image);
};
