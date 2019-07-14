#include "classificator.h"
#include <opencv2/opencv.hpp>
DnnClassificator::DnnClassificator(string path_to_model, string path_to_config, string path_to_labes, int inputWidth, int inputHeight, Scalar mean, int flag_swapRB)
{
	width = inputWidth;
	height = inputHeight;
	this->path_to_config= path_to_config;
	this->path_to_labes= path_to_labes ;
	this->path_to_model= path_to_model;
	//mean = Scalar(0, 0, 0, 0);
	this->swapRB = flag_swapRB;
	Net net = readNet(path_to_model, path_to_config);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);
	
};
Mat DnnClassificator::Classify(Mat image)
{
	Mat inputTensor,r;
	Net net = readNet(path_to_model, path_to_config);
	//mean = Scalar(0, 0, 0, 0);
	blobFromImage(image, inputTensor, 1.0, Size(200,200), mean, swapRB, false);
	net.setInput(inputTensor);
	r=net.forward();
	r.reshape(1, 1);
	return r;
}