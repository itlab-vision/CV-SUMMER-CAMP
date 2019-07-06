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
}

Mat DnnClassificator::Classify(Mat image)
{
	return Mat();
}
