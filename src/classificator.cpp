#include "classificator.h"

DnnClassificator::DnnClassificator(string modelPath_, string configPath_, string labelsPath_, int inputWidth_, int inputHeight_, Scalar mean_, bool swapRB_) {
	modelPath = modelPath_;
	configPath = configPath_;
	labelsPath = labelsPath_;
	width = inputWidth_;
	height = inputHeight_;
	mean = mean_;
	swapRB = swapRB_;
	int backendId = DNN_BACKEND_OPENCV;
	int targetId = DNN_TARGET_CPU;
	net = readNet(modelPath, configPath);
	net.setPreferableBackend(backendId);
	net.setPreferableTarget(targetId);
}


Mat DnnClassificator::Classify(Mat image) {
	Mat inputTensor;
	int scale = 1, ddepth = CV_32F;
	blobFromImage(image, inputTensor, scale, Size(width, height), mean, swapRB, false, ddepth);
	net.setInput(inputTensor);
	Mat prob = net.forward();
	prob = prob.reshape(1, 1);
	return prob;
}