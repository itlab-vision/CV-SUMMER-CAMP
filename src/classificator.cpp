#include "classificator.h"

DnnClassificator::DnnClassificator(String pathToModel, String pathToConfing, String pathToLabels,
	int inputWidth, int inputHeight, Scalar mean, bool swapRB): 
	labelsPath(pathToLabels), width(inputWidth), height(inputHeight) {
	this->mean = mean;
	this->swapRB = swapRB;
	net = readNet(pathToModel, pathToConfing);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);
}

Mat DnnClassificator:: Classify(Mat image) {
	Mat inputTensor = blobFromImage(image, 1.0, Size(width, height),mean,swapRB);
	net.setInput(inputTensor);
	Mat prob = net.forward();
	return prob.reshape(1, 1);
}
