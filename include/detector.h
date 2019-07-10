#pragma once
#include <iostream>
#include <string>
#include <fstream>

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
    virtual String DecodeLabels(int n) = 0;
};

class DnnDetector: public Detector {
private:
  string pathToModel;
  string pathToConfig;
  string pathToLabels;
  int inputWidth = 300;
	int inputHeight = 300;
  double scale = 0.007843;
  Scalar mean = Scalar(127.5, 127.5, 127.5);
  bool swapRB = false;
  Net net;

public:
  DnnDetector(string pathToModel, string pathToConfig, string pathToLabels);
  vector<DetectedObject> Detect(Mat img);
  String DecodeLabels(int n);
};
