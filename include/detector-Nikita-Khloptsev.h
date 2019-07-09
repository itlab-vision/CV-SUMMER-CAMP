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

class Detector{
public:
    virtual vector<DetectedObject> Detect(Mat image) = 0;
};

class DnnDetector : public Detector{
private:
    string model;
    string config;
    string label;
    int width;
    int height;
    double scale;
    Scalar mean;
    bool swapRB;
    Net net;
public:
    DnnDetector(string model, string config, string label, int width, int height, double scale, Scalar mean, bool swapRB);
    vector<DetectedObject> Detect (Mat frame);
};