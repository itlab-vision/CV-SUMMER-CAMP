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
    virtual vector<DetectedObject> Detect(Mat image) = 0 {}
};

class DnnDetector : Detector {
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
    DnnDetector(string pthModel, string pthConfig, string pthLabels, int inputWidth, int inputHeight,
        Scalar myMean = (0, 0, 0, 0), int mySwapRB = 0);
    virtual vector<DetectedObject> Detect(Mat image);


};