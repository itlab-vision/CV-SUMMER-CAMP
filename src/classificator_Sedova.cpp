#include "classificator.h"
#include <opencv2/opencv.hpp>
DnnClassificator::DnnClassificator(string path_to_model, string path_to_config, string path_to_labes, int inputWidth, int inputHeight, vector<int> mean1, int flag_swapRB)
{
	width = inputWidth;
	height = inputHeight;
	path_to_config = this->path_to_config;
	path_to_labes = this->path_to_labes;
	path_to_model = this->path_to_model;
	Net net = readNet(path_to_model, path_to_config);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);
	
};
Mat DnnClassificator::Classify(Mat image)
{
	Mat inputTensor;
	Net net = readNet(path_to_model, path_to_config);
	blobFromImage(image, inputTensor, 1.0, Size(width, height), mean, swapRB, false);
	net.setInput(inputTensor);
	inputTensor=net.forward();
	return inputTensor;
}