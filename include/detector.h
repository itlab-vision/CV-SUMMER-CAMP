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
private:
    vector<string> classesNames;
protected:
    void SetLabels(string labelsPath);
public:
    virtual vector<DetectedObject> Detect(Mat image) = 0 {};
    vector<string> GetLabels();
};

class DnnDetector : public Detector
{
private:
    String modelPath;
    String configPath;
    String labelsPath;
    float scale;
    int inputWidth;
    int inputHeight;
    Scalar mean;
    bool swapRB;
    Net net;
    float scoreThreshold;
public:
    DnnDetector(String modelPath, String configPath, String labelsPath, float scale, int inputWidth, int inputHeight, Scalar mean = (0, 0, 0, 0), bool swapRB = false, float scoreThreshold = 0.75);
    vector<DetectedObject> Detect(Mat image);
};
