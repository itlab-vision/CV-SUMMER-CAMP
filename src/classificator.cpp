#include "classificator.h"

DnnClassificator::DnnClassificator(string _path_to_model, string _path_to_confing, string _path_to_labels,
	int _inputWidth, int _inputHeight, Scalar _mean, bool _swapRB)
{
	path_to_model = _path_to_model;
	path_to_confing = _path_to_confing;
	path_to_labels = _path_to_labels;
	inputWidth = _inputWidth;
	inputHeight = _inputHeight;
	mean = _mean;
	swapRB = _swapRB;

	net = readNet(path_to_model, path_to_confing);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);
}


Mat DnnClassificator::Classify(Mat image)
{
	Mat inputTensor;
	blobFromImage(image, inputTensor, 1, Size(inputWidth, inputHeight), mean, swapRB, false);
	net.setInput(inputTensor);
	Mat prob = net.forward();
	//prob.reshape(1, 1);

	return prob;
}