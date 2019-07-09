#include "classificator.h"

DnnClassificator::DnnClassificator(string path_to_model, string path_to_config, string path_to_labels, int inputWidth, int inputHeight, Scalar mean, bool swapRB) :
	width(inputWidth), height(inputHeight) {
	pathLabels = path_to_labels;
	this->mean = mean;
	this->swapRB = swapRB;
	net = readNet(path_to_model, path_to_config);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);
}


Mat DnnClassificator::Classify(Mat image)
{
	Mat inputTensor;
	blobFromImage(image, inputTensor, 1.0, Size(width, height), mean, swapRB,false);
	net.setInput(inputTensor);
	Mat prob = net.forward();

	
	return prob.reshape(1, 1);

};
