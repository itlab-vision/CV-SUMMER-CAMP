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

class DnnClassificator : public Classificator
{
private:
    string pathToModel;
    string pathToConfig;
    string pathToLabels;
    int imgWidth;
    int imgHeight;
    Scalar mean;
    bool swapRB;
    Net net;
    
public:
    DnnClassificator(string pathToModel, string pathToConfig, string pathToLabels,
                     int imgWidth, int imgHeight, Scalar mean, int swapRB);
    
    Mat Classify(Mat image);
    String DecodeLabel(int n);
    
};

