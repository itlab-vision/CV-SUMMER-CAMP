#include "detector.h"
DnnDetector::DnnDetector(string path_to_model,
	string path_to_config,
	string path_to_labels) {
	this->path_to_model = path_to_model;
	this->path_to_config = path_to_config;
	this->path_to_labels = path_to_labels;

	Net net = readNet(this->path_to_model, this->path_to_config);
	int backendId = DNN_BACKEND_OPENCV;
	int targetId = DNN_TARGET_CPU;
	net.setPreferableBackend(backendId);
	net.setPreferableTarget(targetId);
	this->net = net;
}


vector<DetectedObject> DnnDetector::Detect(Mat image) {
	Mat inputTensor;
	blobFromImage(image, inputTensor, scale, Size(input_width, input_height), mean, swapRB, false);
	net.setInput(inputTensor);
	Mat output = net.forward().reshape(1, 1);
	output = output.reshape(1, output.cols / 7);
	vector<DetectedObject> vecobj;
	for (int i = 0; i < output.rows; i++)
	{
		vecobj.push_back(DetectedObject((int)output.at<float>(i, 1),
										this->DecodeLabel((int)output.at<float>(i, 1) - 1),
										output.at<float>(i, 2), 
										output.at<float>(i, 3)*image.cols,
										output.at<float>(i, 4)*image.rows,
										output.at<float>(i, 5)*image.cols,
										output.at<float>(i, 6)*image.rows));
	}
	return vecobj;
	
}


string DnnDetector::DecodeLabel(int n) {
	ifstream file_labels(path_to_labels);
	vector<string> classesNames;
	string line;
	while (getline(file_labels, line)) {
		classesNames.push_back(line);
	}
	return classesNames[n];
}