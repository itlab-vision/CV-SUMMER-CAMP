#include "classificator.h"

DnnClassificator::DnnClassificator(String path_to_model, String path_to_config, String path_to_labels, int inputWidth, int inputHeight, Scalar mean, bool swapRB)
{
	this->path_to_model = path_to_model;
	this->path_to_config = path_to_config;
	this->path_to_labels = path_to_labels;
	this->inputWidth = inputWidth;
	this->inputHeight = inputHeight;
	this->mean = mean;
	this->swapRB = swapRB;

	this->net= readNet(path_to_model, path_to_config);
	//net.setPreferableBackend();
	//net.setPreferableTarget();
}

Mat DnnClassificator::Classify(Mat image)
{
	double scale = 1.0;
	//int width = 227;
	//int height = 227;

	Mat inputTensor;
	blobFromImage(image, inputTensor, scale, Size(inputWidth, inputHeight), mean, swapRB /* ,0 ,0 */ );
	net.setInput(inputTensor);
	Mat result = net.forward();
	
	/*
	Point classIdPoint;
	double probability;
	minMaxLoc(result.reshape(1, 1), 0, &probability, 0, &classIdPoint);
	int classId = classIdPoint.x;
	*/

	return result;
}

