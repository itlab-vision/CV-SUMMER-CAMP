#include "classificator.h"

DnnClassificator::DnnClassificator(string _model, string _config, string _labels, double _scale, float _inputWidth, float _inputHeight, Scalar _mean, bool _swapRB) {
	model = _model;
	config = _config;
	labels = _labels;
	scale = _scale;
	inputWidth = _inputWidth;
	inputHeight = _inputHeight;
	mean = _mean;
	swapRB = _swapRB;
	net = readNet(model, config);
	net.setPreferableBackend(0);
	net.setPreferableTarget(0);

}
Mat DnnClassificator::Classify(Mat image) {
	Size spatial_size = Size(inputWidth, inputHeight);

	Mat inputTensor;
	blobFromImage(image, inputTensor, scale, spatial_size, mean, swapRB, false, CV_32F);

	//string imgName = "C:/Users/temp2019/Desktop/CV-SUMMER-CAMP/data/lobachevsky.jpg";
	//Mat image1 = imread(imgName);
	//blobFromImage(image1, inputTensor, 1, spatial_size, { 103.94,116.78,123.68 }, false);

	net.setInput(inputTensor);
	Mat prob = net.forward();
	return prob;
	//return Mat();
}
