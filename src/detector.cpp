#include "detector.h"
DnnDetector::DnnDetector(string _model, string _config, string _labels, int _inputWidth, int _inputHeight, Scalar _mean, bool _swapRB)
{
	model = _model;
	config = _config;
	labels = _labels;
	inputWidth = _inputWidth;
	inputHeight = _inputHeight;
	mean = _mean;
	swapRB = _swapRB;
	net = readNet(model, config);

	int backendId = DNN_BACKEND_OPENCV;
	int targetId = DNN_TARGET_CPU;

	net.setPreferableBackend(backendId);
	net.setPreferableTarget(targetId);
}

vector <DetectedObject> DnnDetector::Detect(Mat image) 
{
	Mat inputTensor;
	int scale = 1;
	blobFromImage(image,inputTensor,scale,Size(inputWidth,inputHeight), mean,swapRB,c)
	return image;
}