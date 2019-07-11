#include "classificator.h"

DnnClassificator::DnnClassificator(String path_to_model,
									String path_to_config,
									String path_to_labels,
									int input_width,
									int input_height, bool swapRB,
									Scalar mean) {
	this->path_to_model = path_to_model;
	this->path_to_config = path_to_config;
	this->path_to_labels = path_to_labels;
	this->input_width = input_width;
	this->input_height = input_height;
	this->mean = mean;
	this->swapRB = swapRB;

	Net net = readNet(this->path_to_model, this->path_to_config);
	int backendId = DNN_BACKEND_OPENCV;
	int targetId = DNN_TARGET_CPU;
	net.setPreferableBackend(backendId);
	net.setPreferableTarget(targetId);
	this->net = net;
}


Mat DnnClassificator::Classify(Mat image) {
	Mat inputTensor;
	blobFromImage(image, inputTensor, 1.0, Size(input_width, input_height), mean, swapRB, false);
	net.setInput(inputTensor);
	return net.forward().reshape(1, 1);
}


String DnnClassificator::DecodeLabel(int n) {
	ifstream file_labels(path_to_labels);
	String line;
	while (getline(file_labels, line)) {
		classesNames.push_back(line);
	}
	return classesNames[n];
}



