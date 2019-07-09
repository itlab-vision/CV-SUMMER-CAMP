#include "classificator.h"

DnnClassificator::DnnClassificator(string _model, string _config, string _labels, int _inputWidth, int _inputHeight, Scalar _mean, bool _swapRB)
{
	model = _model;
	config = _config;
	labels = _labels;
	inputWidth = _inputWidth;
	inputHeight = _inputHeight;
	mean = _mean;
	swapRB = _swapRB;

	int backendId = DNN_BACKEND_OPENCV;
	int targetId = DNN_TARGET_CPU;

	//load net
	net = readNet(model, config);
	net.setPreferableBackend(backendId);
	net.setPreferableTarget(targetId);
}

 Mat DnnClassificator::Classify(Mat image)
{
	 Mat inputTensor;
	 double scale = 1;
	 blobFromImage(image, inputTensor, scale, Size(inputWidth,inputHeight), mean, swapRB);
	 net.setInput(inputTensor);
	 Mat prob = net.forward();
	 prob = prob.reshape(1, 1);
	 return prob;
}