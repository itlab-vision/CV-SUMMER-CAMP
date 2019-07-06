#pragma once
#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "detectedobject.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;

class Detector
{
public:
    virtual vector<DetectedObject> Detect(Mat image) = 0;
};

class DnnDetector: public Detector
{
    vector<string> classes;
    string model, config;
    int inputHeight, inputWidth;
    Scalar mean;
    int backendId, targetId;
    bool swapRB;
    Net net;
    double scale;
    //double confThreshold = 0.5;
    
public:
    DnnDetector(vector<string> classes, string model, string config, int inputWidth = 300, int inputHeight = 300, bool swapRB = false, double scale = 0.007843, Scalar mean = {127.5,127.5,127.5}, int backendId = DNN_BACKEND_OPENCV, int targetId = DNN_TARGET_CPU);
    
    vector<DetectedObject> Detect(Mat image);
};
