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
    string model, config, labels;
    int width, height, swapRB;
    Scalar mean;
    Net net;
    int backendId;
    int targetId;
    Mat blob;
    double scale;
    int ddepth;
    bool crop;


public:
    DnnClassificator(string pthModel, string pthConfig, string pthLabels, int inputWidth, int inputHeight, 
        Scalar myMean = (0, 0, 0, 0), int mySwapRB = 0);
   virtual Mat Classify(Mat image);

};