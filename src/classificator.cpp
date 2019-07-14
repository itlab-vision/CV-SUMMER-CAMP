#include "classificator.h"

DnnClassificator::DnnClassificator(String path_to_model, String path_to_config, String  path_to_lables, int width, int height, Scalar mean, bool swapRB)
{
	this->path_to_config = path_to_config;
	this->path_to_lables = path_to_lables;
	this->path_to_model = path_to_model;
	this->spatial_size = Size(width, height);
	this->mean = mean;
	this->swapRB = swapRB;
	net = readNet(path_to_model, path_to_config);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
}

Mat DnnClassificator::Classify(Mat image)
{
	Mat inputTensor;
	double scale = 1;
	int ddepth = CV_32F;

	blobFromImage(image, inputTensor, scale, spatial_size, mean, swapRB,false,ddepth);
	net.setInput(inputTensor);
	Mat prob = net.forward().reshape(1,1);
	return prob;
}
