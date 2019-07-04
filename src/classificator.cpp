#include "classificator.h"

DnnClassificator::DnnClassificator(string _pathToModel, string _pathToConfig, string _pathToLabels, int _inputWidth, int _inputHeight, Scalar _mean, bool _swapRB)
{
	pathToModel = _pathToModel;
	pathToConfig = _pathToConfig;
	pathToLabels = _pathToLabels;
	inputWidth = _inputWidth;
	inputHeight = _inputHeight;
	mean = _mean;
	swapRB = _swapRB;

	net = readNet(pathToModel, pathToConfig);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);
};
Mat DnnClassificator::Classify(Mat frame)
{
	Mat inputTensor;
	blobFromImage(frame, inputTensor, 1, Size(inputWidth, inputHeight), mean, swapRB, false, CV_32F);
	net.setInput(inputTensor);
	Mat prob = net.forward();
	prob = prob.reshape(1, 1);
	return prob;
};