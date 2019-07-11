#include "classificator.h"


DnnClassificator::DnnClassificator(string caffemodel, string prototxt, string labels, int inputWidth, int inputHeight, Scalar mean, bool swapRB)
{
	this->caffemodel = caffemodel;
	this->prototxt = prototxt;
	this->labels = labels;
	this->inputWidth = inputWidth;
	this->inputHeight = inputHeight;
	this->mean = mean;
	this->swapRB = swapRB;

	Net net = readNet(caffemodel, prototxt);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	this->net = net;
}

Mat DnnClassificator::Classify(Mat image)
{
	Mat inputTensor;
	blobFromImage(image, inputTensor, 1, Size(inputWidth, inputHeight), mean, swapRB, false);
	net.setInput(inputTensor);
	Mat output = net.forward();
	return output.reshape(1, 1);

}