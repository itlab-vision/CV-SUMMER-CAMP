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

class DnnClassificator: public Classificator
{
    string model, config, labels;
    int inputHeight, inputWidth;
    Scalar mean;
    int backendId, targetId;
    bool swapRB;
    Net net;
    double scale;
public:
    DnnClassificator(string path_model, string path_config, string path_labels,
                     int inpw, int inph, bool sw = false, double sc = 1.0,
                     Scalar m = {0,0,0,0},int back = DNN_BACKEND_OPENCV, int targ = DNN_TARGET_CPU);
    Mat Classify(Mat image);
    
    
};
