#include "classificator.h"

DnnClassificator::DnnClassificator(string path_to_model, string path_to_config, string path_to_labels, int _inputWidth, int _inputHeight, bool _swapRB, Scalar _mean)
{
	model_path = path_to_model;
	config_path = path_to_config;
	labels_path = path_to_labels;
	inputWidth = _inputWidth;
	inputHeight = _inputHeight;
	swapRB = _swapRB;
	mean = _mean;
	net = readNet(model_path, config_path);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
}

Mat DnnClassificator::Classify(Mat image)
{
	Mat inputTensor;
	blobFromImage(image, inputTensor, 1.0, Size(inputWidth, inputHeight), mean, swapRB, false);
	Net net;
	net.setInput(inputTensor);
	Mat prob = net.forward();
	prob=prob.reshape(1, 1);
	return prob;
}