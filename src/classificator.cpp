#include "classificator.h"



DnnClassificator::DnnClassificator(const String & ptm, const String & ptc, uint32_t width, uint32_t height, const Scalar & mean, bool swapRB) {
	net = readNet(ptm, ptc);

	net.setPreferableBackend(0);
	net.setPreferableTarget(0);

	this->width = width;
	this->height = height;
	this->mean = mean;
	this->swapRB = swapRB;
}


Mat DnnClassificator::Classify(Mat image) {
	Mat inputTensor;

	blobFromImage(image, inputTensor, 1.0, { (int)width, (int)height }, mean, swapRB, false);
	net.setInput(inputTensor);

	return net.forward().reshape(1, 1);
}
