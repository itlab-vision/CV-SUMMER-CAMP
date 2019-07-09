#include "classificator.h"


DnnClassificator::DnnClassificator(string model_path, string config_path, int inputWidth, int inputHeight, bool swapRB, Scalar scalar) {
	net = readNet(model_path, config_path);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);

	this->width = inputWidth;
	this->height = inputHeight;
	this->swapRB = swapRB;
	this->scalar = scalar;
}


Mat DnnClassificator::Classify(Mat image) {
	Mat input;
	blobFromImage(image, input, 1.0, { width, height }, scalar, swapRB, false);
	net.setInput(input);
	return net.forward().reshape(1, 1);
}
