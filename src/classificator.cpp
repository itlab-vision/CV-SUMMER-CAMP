#include "classificator.h"

DnnClassificator::DnnClassificator(string model_path, string config_path, string labels_path, int inputWidth, int inputHeight, bool mirror, Scalar scalar = (0,0,0)): 
	model_path(model_path), config_path(config_path), labels_path(labels_path), width(inputWidth), height(inputHeight), mirror(mirror)
{
	this->scalar = scalar;
	loadLabels();
	net = readNet(model_path, config_path);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);
}
void DnnClassificator::loadLabels()
{
	string input;
	ifstream in(labels_path);
	while (getline(in, input))
	{
		classesNames.push_back(input);
	}
}
vector<string> DnnClassificator::getClassesNames()
{
	return classesNames;
}

Mat DnnClassificator::Classify(Mat image)
{
	Mat result;
	Mat inputTensor;
	blobFromImage(image, inputTensor, 1.0, Size(width, height), scalar, mirror, false);
	net.setInput(inputTensor);
	result = net.forward();
	result = result.reshape(1, 1);

	return result;
}