#include "classificator.h"
#include "filter.h"

DnnClassificator::DnnClassificator(String _modelPath, String _configPath, String _labelsPath, int _inputWidth, int _inputHeight, Scalar _mean, bool _swapRB) {
	modelPath = _modelPath;
	configPath = _configPath;
	labelsPath = _labelsPath;
	inputWidth = _inputWidth;
	inputHeight = _inputHeight;
	mean = _mean;
	swapRB = _swapRB;

	net = readNet(modelPath, configPath);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);
}

Mat DnnClassificator::Classify(Mat image) {
	
	Mat inputTensor;
	blobFromImage(image, inputTensor, /** scale = **/  1, Size(inputWidth, inputHeight), mean, swapRB, false);



	net.setInput(inputTensor);
	Mat prob = net.forward();

	return prob.reshape(1, 1);

}