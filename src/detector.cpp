#include "detector.h"

DnnDetector::DnnDetector(string _modelPath, string _configPath, string _labelPath, int _width, int _height, Scalar _mean, bool _swapRB)
{
	modelPath = _modelPath;
	configPath = _configPath;
	labelPath = _labelPath;
	width = _width;
	height = _height;
	mean = _mean;
	swapRB = _swapRB;
	net = readNet(_modelPath, _configPath);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);
};
vector<DetectedObject> DnnDetector::Detect(Mat frame)
{
	Mat inputTensor;
	blobFromImage(frame, inputTensor, 1, Size(width, height), mean, swapRB, false, CV_32F);
	net.setInput(inputTensor);
	Mat prob = net.forward();
	prob = prob.reshape(1, 100);
	return prob;
};