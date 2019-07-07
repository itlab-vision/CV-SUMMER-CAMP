#include "detector.h"

DnnDetector::DnnDetector(string _modelPath, string _configPath, string _labelPath, int _width, int _height, double _scale, Scalar _mean, bool _swapRB)
{
	modelPath = _modelPath;
	configPath = _configPath;
	labelPath = _labelPath;
	width = _width;
	height = _height;
	scale = _scale;
	mean = _mean;
	swapRB = _swapRB;
	net = readNet(_modelPath, _configPath);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
};
vector<DetectedObject> DnnDetector::Detect(Mat frame)
{
	Mat inputTensor;
	blobFromImage(frame, inputTensor, scale, Size(width, height), mean, swapRB, false);
	net.setInput(inputTensor);
	Mat prob = net.forward();
	prob = prob.reshape(1, 1);
	int n = prob.cols / 7;
	prob = prob.reshape(1, n);
	vector<DetectedObject> objects;
	int cols = frame.cols;
	int rows = frame.rows;
	for (int i = 0; i < n; i++) {
		DetectedObject tmp;
		tmp.classid = prob.at<float>(i, 1);
		tmp.score = prob.at<float>(i, 2);
		tmp.Left = prob.at<float>(i, 3) * cols;
		tmp.Bottom = prob.at <float>(i, 4) * rows;
		tmp.Right = prob.at<float>(i, 5) * cols;
		tmp.Top = prob.at<float>(i, 6) * rows;
		objects.push_back(tmp);
	return objects;
};