#include "classificator.h"


DnnClassificator::DnnClassificator(string _modelPath, string _configPath, string _labelsPath,
	int inputWidth, int inputHeight, Scalar _mean, bool _swapRB) {
	
	modelPath = _modelPath;
	configPath = _configPath;
	labelsPath = _labelsPath;
	width = inputWidth;
	height = inputHeight;
	mean = _mean;
	swapRB = _swapRB;

	int backendId = DNN_BACKEND_OPENCV;
	int targetId = DNN_TARGET_CPU;

    net = readNet(modelPath, configPath);
	net.setPreferableBackend(backendId);
	net.setPreferableTarget(targetId);
}

Mat DnnClassificator::Classify(Mat image) {
	Mat inputTensor;
	int scale = 1;
	int ddepth = CV_32F;
	blobFromImage(image, inputTensor, scale, Size(width, height), mean, swapRB, false, ddepth);
	net.setInput(inputTensor);
	

	return net.forward();
}